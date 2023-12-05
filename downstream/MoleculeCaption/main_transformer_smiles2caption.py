import datetime
#Based on https://pytorch.org/tutorials/beginner/translation_transformer.html

from lib2to3.pgen2 import token
import torch_geometric
import torch
from torch.nn import DataParallel
from transformers import AutoTokenizer, LogitsProcessorList, BeamSearchScorer, BertTokenizer, T5Tokenizer
from torch import nn, Tensor
import torch.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from torch.utils.data.dataloader import default_collate

from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np

import pickle

import argparse
import sys

from model.GinT5 import GinDecoder, tensor_to_str
# from model.GinGPT import GinDecoder, ClipCaptionModel, ClipCaptionPrefix
from model.gin_model import GNN
from dataloader import TextMoleculeReplaceDataset, custom_collate
from tqdm import tqdm

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
parser.add_argument('--nlayers', type=int, default=6, help='number of layers')
parser.add_argument('--emb_size', type=int, default=512, help='input dimension size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--nhead', type=int, default=8, help='num attention heads')

parser.add_argument('--MoMuK', default=False, action='store_true')
parser.add_argument('--model_size', type=str, default='base')
parser.add_argument('--data_path', type=str, default='data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='saved_models/', help='path where weights are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--use_scheduler', type=bool, default=True, help='Use linear scheduler')
parser.add_argument('--num_warmup_steps', type=int, default=400, help='Warmup steps for linear scheduler, if enabled.')

parser.add_argument('--output_file', type=str, default='out.txt', help='path where test generations are saved')
parser.add_argument('--use_3d', default=False, action='store_true')
parser.add_argument('--test_ckpt', type=str, default='')

args = parser.parse_args()

runseed = 100
torch.manual_seed(runseed)
np.random.seed(runseed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(runseed)


tokenizer = T5Tokenizer.from_pretrained("molt5-"+args.model_size+"-smiles2caption/", model_max_length=512)

train_data = TextMoleculeReplaceDataset(args.data_path, 'train', tokenizer)
val_data = TextMoleculeReplaceDataset(args.data_path, 'validation', tokenizer)
test_data = TextMoleculeReplaceDataset(args.data_path, 'test', tokenizer)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=20, collate_fn=custom_collate, pin_memory=True)

# if args.MoMuK:
#     print("model init with MoMu-K")
# else:
#     print("model init with MoMu-S")
print('model init with MoLM')

# my_model = DataParallel(GinDecoder(has_graph=True, MoMuK=args.MoMuK, model_size=args.model_size, use_3d=args.use_3d).to(device))
my_model = GinDecoder(has_graph=True, MoMuK=args.MoMuK, model_size=args.model_size, use_3d=args.use_3d).to(device)

if args.mode == 'test':
    state_dict = torch.load(args.test_ckpt)
    (my_model.module if hasattr(my_model, 'module') else my_model).load_state_dict(state_dict)

if args.mode == 'train':
    for p in my_model.named_parameters():
    	if p[1].requires_grad:
            print(p[0])

    pg = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr)
    # num_training_steps = args.epochs * len(train_dataloader) - args.num_warmup_steps
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_training_steps) 

MAX_LENGTH = args.max_length


def train_epoch(dataloader, model, optimizer, epoch):
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()


    # model.train()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    failed = 0
    for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            # model.zero_grad()

            graph = d['graph']
            for idx, val in graph.items():
                if hasattr(val, 'to'):
                    graph[idx] = val.to(device)

            smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask

            text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            label = text_tokens_['input_ids'].to(device)  # caption

            label = label.masked_fill(~text_mask.bool(), -100)

            # print(smiles_tokens)
            # print(src_padding_mask)
            # print(label)
            # print(text_mask)

            loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)

            if j % 300 == 0:
                print('total steps: {}, step: {}, loss: {}'.format(epoch * len(dataloader) + j, j, loss))

            # print('Loss: ' + str(loss))

            # For multi-GPU
            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()

            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            losses += loss.item()
        except Exception as e:
            print(f'Skipped step due to {e}')
            failed += 1

    overall_loss = losses / (len(dataloader) - failed)
    wandb.log({"train_loss": overall_loss, "epoch": i})
    wandb.log({"failed_steps": failed, "epoch": i})
    return overall_loss


def eval(dataloader, model, epoch):
    model.eval()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):

            graph = d['graph']
            for idx, val in graph.items():
                if hasattr(val, 'to'):
                    graph[idx] = val.to(device)

            smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            label = text_tokens_['input_ids'].to(device)  # caption

            label = label.masked_fill(~text_mask.bool(), -100)

            loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)
            losses += loss.item()
            if j % 100 == 0:
                print('val total steps: {}, step: {}, val loss: {}'.format(epoch*len(dataloader) + j, j, loss))
    
    return losses/len(dataloader)


if args.mode == 'train':
    # my_model.train()
    min_val_loss = 10000
    modal = ('3d' if args.use_3d else '2d')
    wandb.init(project='smiles2caption', name=f'{str(datetime.datetime.now())} {args.model_size} ({modal.upper()})')
    wandb.watch(my_model)
    for i in range(args.epochs):
        print('Epoch:', i)
        train_epoch(train_dataloader, model=my_model, optimizer=optimizer, epoch=i)
        val_loss = eval(val_dataloader, model=my_model, epoch=i)
        wandb.log({"val_loss": val_loss, "epoch": i})
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("--------------------save model--------------------")
            torch.save((my_model.module if hasattr(my_model, 'module') else my_model).state_dict(),
                       args.saved_path + 'gint5_smiles2caption_' + args.model_size + '_' + modal + '.pt')


if args.mode == 'test':
    # Fix gather


    my_model.eval()
    smiles = []
    test_outputs = []
    test_gt = []
    with torch.no_grad():
        for j, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            real_text = d['description']
            graph = d['graph']
            for idx, val in graph.items():
                if hasattr(val, 'to'):
                    graph[idx] = val.to(device)

            smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            # text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            # text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            # label = text_tokens_['input_ids'].to(device)  # caption
          
            outputs = my_model(graph, smiles_tokens, src_padding_mask, tokenizer)
            # output_strs = []
            # if isinstance(outputs, list):
            #     for outputs_real in outputs:
            #         output_strs.extend(tokenizer.batch_decode(outputs_real, skip_special_tokens=True))
            # else:
            #     output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            smiles.extend(d['smiles'])
            test_gt.extend(real_text)
            # print(output_strs)
            # test_outputs.extend(output_strs)
            test_outputs.extend(outputs)
            
    with open(args.output_file, 'w') as f:
        f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for smi, rt, ot in zip(smiles, test_gt, test_outputs):
            f.write(smi + '\t' + rt + '\t' + ot + '\n')

# def test_eval(dataloader, model):
#     model.eval()
#     smiles = []
#     test_outputs = []
#     test_gt = []
#     with torch.no_grad():
#         for j, d in enumerate(dataloader):
#             if j % 100 == 0: print('Test Step:', j)
#             graph = d['graph'].to(device)
#             labels = d['text'].to(device)
#             labels[labels == tokenizer.pad_token_id] = -100
#             print(model.translate(graph, labels, tokenizer))
#             # real_text = d['description']
#             # smiles.extend(d['smiles'])
#             # test_gt.extend(real_text)
            
#             # test_outputs.extend([graph2caption(model, graph, tokenizer) for smi in d['smiles']])

#             #wandb.log({'test total steps':len(dataloader) + j, 'step':j,'test loss' : loss})

#     return smiles, test_gt, test_outputs

# smiles, test_gt, test_outputs = test_eval(test_dataloader, model)


#wandb.finish()
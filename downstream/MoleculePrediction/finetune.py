import argparse
import importlib
import sys

from torch.utils.data import DataLoader

from loader import MoleculeDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
import re
import wandb

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

from tensorboardX import SummaryWriter

sys.path.insert(0, '../graph-transformer')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for idx, val in batch.items():
            if hasattr(val, 'to'):
                batch[idx] = val.to(device)
        pred = model(batch)
        y = batch['y'].view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch = batch.to(device)
        for idx, val in batch.items():
            if hasattr(val, 'to'):
                batch[idx] = val.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch['y'].view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


# def match_custom_collate(data):
#     graphs = []
#     texts = []
#     masks = []
#     for graph, text, mask in data:
#         graphs.append(data_to_graph(graph))
#         texts.append(text)
#         masks.append(mask)
#
#     # Collate 3d the graphs
#     max_node = 512
#     multi_hop_max_dist = 5
#     spatial_pos_max = 1024
#     number_graph_list(graphs)
#     collated_graphs = collator_3d(graphs, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
#                                   spatial_pos_max=spatial_pos_max)
#     return collated_graphs, torch.stack(texts), torch.stack(masks)

def number_graph_list(graph_list):
    for i, graph in enumerate(graph_list):
        graph.idx = i


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def data_to_graph(data):
    new_graph = AttrDict()
    new_graph.update(data.to_dict())
    new_graph = preprocess_item(new_graph)
    return new_graph


def custom_collate(data):
    graphs = [data_to_graph(d) for d in data]
    max_node = 512
    multi_hop_max_dist = 5
    spatial_pos_max = 1024
    number_graph_list(graphs)
    return collator_3d(graphs, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                                      spatial_pos_max=spatial_pos_max)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0.00,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--use_3d', default=False, action='store_true')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    #tox21, hiv, pcba, muv, bace, toxcast, sider, clintox
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

 
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=custom_collate)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, use_3d=args.use_3d)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.transformer.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_head.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    # Random forest code
    def read_dataloader(dataloader):
        model.eval()
        embeddings = []
        labels = []
        for collated in tqdm(dataloader):
            for idx, val in collated.items():
                if hasattr(val, 'to'):
                    collated[idx] = val.to(device)
            graph_embeddings = model.forward_through_graph_encoder(collated)
            embeddings.extend([t.numpy() for t in torch.unbind(graph_embeddings.detach().cpu(), dim=0)])
            prediction_labels = [t.numpy().item() for t in torch.unbind(collated['y'].detach().cpu(), dim=0)]
            prediction_labels = [1 if i == 1 else 0 for i in prediction_labels]
            labels.extend(prediction_labels)
            # print(labels)

        return embeddings, labels

    train_embeddings, train_labels = read_dataloader(train_loader)
    val_embeddings, val_labels = read_dataloader(val_loader)
    test_embeddings, test_labels = read_dataloader(test_loader)

    # param = {'max_depth': 3, 'eta': 0.3, 'objective': 'binary:logistic', 'nthread': -1, 'eval_metric': 'logloss'}
    # dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    # dtest = xgb.DMatrix(test_embeddings, label=test_labels)
    #
    # num_round = 50
    # bst = xgb.train(param, dtrain, num_round)
    #
    # # Make predictions on the test set
    # preds = bst.predict(dtest)
    # preds = [1 if p > 0.5 else 0 for p in preds]
    #
    # xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                             colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
    #                             max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
    #                             n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
    #                             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
    #                             silent=False, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
    # xgb_gbc.fit(train_embeddings, train_labels, eval_set=[(val_embeddings, val_labels)], eval_metric='auc', early_stopping_rounds=300)
    # pre_pro = xgb_gbc.predict_proba(test_embeddings)[:, 1]
    # fpr, tpr, threshold = roc_curve([float(i) for i in test_labels], pre_pro)
    # AUC = auc(fpr, tpr)
    #
    # model_filename = "xgb_trained_model.model"
    # xgb_gbc.save_model(model_filename)
    #
    # # Evaluate the performance
    # print(f"Accuracy: {AUC}")

    wandb.init(project=f'MoleculePrediction-{args.dataset}', config=args)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = round(eval(args, model, device, test_loader), 4)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        wandb.log({'val_acc': val_acc, 'test_acc': test_acc, 'epoch': epoch})

        if epoch > 1:
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

        print("")

        if epoch > 1 and test_acc >= max(test_acc_list):
            pretrain_epoch = re.findall(r'\d+', args.input_model_file)[0]
            torch.save(model.state_dict(), "./GIN_checkpoints/"+args.dataset+"_pre"+pretrain_epoch+ "_test" + str(test_acc) +".pth")

        if test_acc_list:
            print(f"CURRENT BEST: {max(test_acc_list)}")


if __name__ == "__main__":
    main()

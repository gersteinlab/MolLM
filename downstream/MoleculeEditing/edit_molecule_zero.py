import csv
import importlib
import sys
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import List

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.transform_zinc250k import zinc250_atomic_num_list
from mflow.models.utils import check_validity, construct_mol

# Initialize MoLM

sys.path.insert(0, '../graph-transformer')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d

args = SimpleNamespace(_name='transformer_m_base', act_dropout=0.3, activation_fn='gelu', add_3d=False,
                       add_prev_output_tokens=False, all_gather_list_size=16384, amp=False,
                       amp_batch_retries=2,
                       amp_init_scale=128, amp_scale_window=None, apply_init=True,
                       arch='transformer_m_base',
                       attention_dropout=0.3, azureml_logging=False, batch_size=256, batch_size_valid=256,
                       best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False,
                       bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0,
                       combine_valid_subsets=None, cpu=False, cpu_offload=False,
                       criterion='graph_prediction',
                       curriculum=0, data_buffer_size=20, data_path='NOT-IN-USE', dataset_impl=None,
                       dataset_name='NOT-IN-USE', ddp_backend='legacy_ddp', ddp_comm_hook='none',
                       device_id=0, disable_validation=False, distributed_backend='nccl',
                       distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=2,
                       distributed_port=-1, distributed_rank=0, distributed_world_size=2, dropout=0.1,
                       droppath_prob=0.1, edge_type='multi_hop', ema_decay=0.9999, ema_fp32=False,
                       ema_seed_model=None, ema_start_update=0, ema_update_freq=1, empty_cache_freq=0,
                       encoder_attention_heads=32, encoder_embed_dim=768, encoder_ffn_embed_dim=768,
                       encoder_layers=12, encoder_learned_pos=True, encoder_normalize_before=True, eos=2,
                       fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None,
                       fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False,
                       fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0,
                       fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test',
                       gradient_as_bucket_view=False, grouped_shuffling=False, heartbeat_timeout=-1,
                       ignore_unused_valid_subsets=False, init_token=None, keep_best_checkpoints=-1,
                       keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1,
                       load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file=None,
                       log_format=None, log_interval=100, lr=[0.25], lr_scheduler='fixed', lr_shrink=0.1,
                       max_epoch=0, max_positions=512, max_tokens=None, max_tokens_valid=None, max_update=0,
                       max_valid_steps=None, maximize_best_checkpoint_metric=False,
                       memory_efficient_bf16=False,
                       memory_efficient_fp16=False, metric='mae', min_loss_scale=0.0001,
                       mode_prob='0.2,0.2,0.6', model_parallel_size=1, multi_hop_max_dist=5, no_2d=False,
                       no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False,
                       no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False,
                       no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False,
                       noise_scale=0.2, not_fsdp_flatten_parameters=False, nprocs_per_node=2,
                       num_3d_bias_kernel=128, num_atoms=4608, num_classes=1, num_edge_dis=128,
                       num_edges=1536,
                       num_in_degree=512, num_out_degree=512, num_segment=2, num_shards=1, num_spatial=512,
                       num_workers=16, on_cpu_convert_precision=False, optimizer=None,
                       optimizer_overrides='{}',
                       pad=1, patience=-1, pipeline_balance=None, pipeline_checkpoint='never',
                       pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None,
                       pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None,
                       pipeline_model_parallel=False, plasma_path='/tmp/plasma',
                       pooler_activation_fn='tanh',
                       profile=False, quantization_config_path=None, required_batch_size_multiple=8,
                       required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False,
                       reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False,
                       restore_file='checkpoint_last.pt', sandwich_ln=False, save_dir='NOT-IN-USE',
                       save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sent_loss=False,
                       sentence_avg=False, sentence_class_num=2, separator_token=None, shard_id=0,
                       share_encoder_input_output_embed=False, shorten_data_split_list='',
                       shorten_method='none', simul_type=None, skip_invalid_size_inputs_valid_test=False,
                       slowmo_base_algorithm='localsgd', slowmo_momentum=None, split='valid',
                       stop_min_lr=-1.0,
                       stop_time_hours=0, store_ema=False, suppress_crashes=False, task='NOT-IN-USE',
                       tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, tpu=False,
                       train_subset='train', unk=3, update_epoch_batch_itr=False, update_freq=[1],
                       update_ordered_indices_seed=False, use_bmuf=False, use_plasma_view=False,
                       use_sharded_state=False,
                       user_dir='NOT-IN-USE',
                       valid_subset='valid', validate_after_updates=0, validate_interval=1,
                       validate_interval_updates=0, wandb_project=None, warmup_updates=0,
                       write_checkpoints_asynchronously=False, zero_sharding='none')

transformerm_model = TransformerM(args=args)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
transformerm_model = transformerm_model.to(device)
transformerm_model.eval()

# Initialize MoFlow
import os
import time
import argparse
from distutils.util import strtobool
from mflow.models.hyperparams import Hyperparameters
from mflow.utils.model_utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, default='This molecule is beautiful.')
parser.add_argument("--checkpoint_name", type=str,
                    default='littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt')
parser.add_argument("--model_dir", type=str, default='./results')
parser.add_argument("--data_dir", type=str, default='../data')
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
# parser.add_argument('--molecule_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz',
#                     help='path to molecule dataset')
parser.add_argument("--snapshot-path", "-snapshot", type=str, required=True)
parser.add_argument("--hyperparams-path", type=str, default='moflow-params.json', required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument('--additive_transformations', type=strtobool, default='false',
                    help='apply only additive coupling layers')
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--n_experiments', type=int, default=1, help='number of times generation to be run')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature of the gaussian distribution')
# parser.add_argument('--draw_neighborhood', type=strtobool, default='true',
#                     help='if neighborhood of a molecule to be visualized')

parser.add_argument('--save_fig', type=strtobool, default='true')
parser.add_argument('--save_score', type=strtobool, default='true')
parser.add_argument('-r', '--reconstruct', action='store_true', default=False)
# parser.add_argument('-i', '--interpolation', action='store_true', default=False)
parser.add_argument('--int2point', action='store_true', default=False)
parser.add_argument('--intgrid', action='store_true', default=False)

parser.add_argument('--inter_times', type=int, default=5)

parser.add_argument('--correct_validity', type=strtobool, default='true',
                    help='if apply validity correction after the generation')
args = parser.parse_args(
    "--model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask -snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k --hyperparams-path moflow-params.json --temperature 0.85 --batch-size 1 --n_experiments 5".split(
        " "))

start = time.time()
print("Start at Time: {}".format(time.ctime()))
# chainer.config.train = False
snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
print("loading hyperparamaters from {}".format(hyperparams_path))
model_params = Hyperparameters(path=hyperparams_path)
moflow_model = load_model(snapshot_path, model_params, debug=True)
if len(moflow_model.ln_var) == 1:
    print('model.ln_var: {:.2f}'.format(moflow_model.ln_var.item()))
elif len(moflow_model.ln_var) == 2:
    print('model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}'.format(moflow_model.ln_var[0].item(),
                                                                    moflow_model.ln_var[1].item()))

moflow_model.to(device)
print(f'device = {device}')
moflow_model.eval()  # Set model for evaluation
for name, param in moflow_model.named_parameters():
    param.requires_grad = False

from mflow.utils.model_utils import smiles_to_adj, rescale_adj


def smiles_to_moflow_rep(smiles):
    global moflow_model, device

    adj, atoms = smiles_to_adj(smiles, 'zinc250k')
    adj_normalized = rescale_adj(adj)
    # with torch.no_grad():
    z0, _ = moflow_model(adj.to(device), atoms.to(device), adj_normalized.to(device))

    h, adj_h = z0
    # Flatten h and adj_h into 1D tensors
    h_flat = h.view(h.shape[0], -1)
    adj_h_flat = adj_h.view(adj_h.shape[0], -1)

    return torch.cat([h_flat, adj_h_flat], dim=1)


#

# Load TransformerM checkpoint
d = torch.load(
    './model-epoch=507.ckpt')
state_dict = d['state_dict']
graph_encoder_state_dict = {}
text_encoder_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('graph_encoder.'):
        graph_encoder_state_dict[k.removeprefix('graph_encoder.')] = v
    elif k.startswith('text_encoder.'):
        text_encoder_state_dict[k.removeprefix('text_encoder.')] = v

transformerm_model.load_state_dict(graph_encoder_state_dict, strict=True)

# Text Encoder
import torch.nn as nn
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = BertModel.from_pretrained('bert_pretrained/')
        else:
            config = BertConfig(vocab_size=31090, )
            self.main_model = BertModel(config)

        self.dropout = nn.Dropout(0.1)
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        typ = torch.zeros(input_ids.shape).long().to(device)
        output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        logits = self.dropout(output)
        return logits


text_encoder = TextEncoder(pretrained=False)
text_encoder.load_state_dict(text_encoder_state_dict, strict=True)
text_encoder.to(device)
text_encoder.eval()

from transformers import BertTokenizer
from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from rdkit.Chem import AllChem, Mol, rdMolDescriptors, Descriptors, QED
from torch_geometric.data import Data
import re
from rdkit import Chem, RDLogger
import numpy as np

atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def construct_mol(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    mol = Chem.AddHs(mol)

    bad = False

    # rdDepictor.Compute2DCoords(mol)
    try:
        if AllChem.EmbedMolecule(mol) == -1:
            bad = True
    except Exception as _:
        pass
    # AllChem.EmbedMolecule(mol)

    mol_try = Chem.Mol(mol)
    if not bad:
        try:
            AllChem.MMFFOptimizeMolecule(mol_try)
            mol = mol_try
        except Exception as _:
            pass

    mol = Chem.RemoveHs(mol)

    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        # atom_feature = [allowable_features['possible_atomic_num_list'].index(
        #     atom.GetAtomicNum())] + [allowable_features[
        #     'possible_chirality_list'].index(atom.GetChiralTag())]
        # atom_features_list.append(atom_feature)
        atom_features_list.append(atom_to_feature_vector(atom))

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    # positions
    try:
        if not bad:
            positions = mol.GetConformer().GetPositions()
        else:
            num_atoms = mol.GetNumAtoms()
            positions = np.zeros((num_atoms, 3))
    except ValueError:
        return None
    # bonds
    # num_bond_features = 2   # bond type, bond direction
    # if len(mol.GetBonds()) > 0: # mol has bonds
    #     edges_list = []
    #     edge_features_list = []
    #     for bond in mol.GetBonds():
    #         i = bond.GetBeginAtomIdx()
    #         j = bond.GetEndAtomIdx()
    #         edge_feature = [allowable_features['possible_bonds'].index(
    #             bond.GetBondType())] + [allowable_features[
    #                                         'possible_bond_dirs'].index(
    #             bond.GetBondDir())]
    #         edges_list.append((i, j))
    #         edge_features_list.append(edge_feature)
    #         edges_list.append((j, i))
    #         edge_features_list.append(edge_feature)

    #     # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    #     edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    #     # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #     edge_attr = torch.tensor(np.array(edge_features_list),
    #                              dtype=torch.long)
    # else:   # mol has no bonds
    #     edge_index = torch.empty((2, 0), dtype=torch.long)
    #     edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=torch.from_numpy(edge_index).to(torch.int64),
                edge_attr=torch.from_numpy(edge_attr).to(torch.int64))

    data.__num_nodes__ = len(x)
    data.pos = positions

    return data


def MolTransfer(x, adj, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    adj = adj.detach().cpu()
    x = x.detach().cpu()
    mol = construct_mol(x, adj, atomic_num_list)

    return mol, mol_to_graph_data_obj_simple(mol)


def generate_mols_fix(model, atomic_num_list, temp=0.7, z_mu=None, batch_size=20, true_adj=None):  # gpu=-1):
    global device
    """

    :param model: Moflow model
    :param z_mu: latent vector of a molecule
    :param batch_size:
    :param true_adj:
    :param gpu:
    :return:
    """

    adj, x = model.reverse(z_mu, true_adj=true_adj)
    return adj, x


import torch.nn.functional as F


def tokenizer_text(text, text_max_len):
    tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
    sentence_token = tokenizer(text=text,
                               truncation=True,
                               padding='max_length',
                               add_special_tokens=False,
                               max_length=text_max_len,
                               return_tensors='pt',
                               return_attention_mask=True)
    input_ids = sentence_token['input_ids']
    attention_mask = sentence_token['attention_mask']
    return input_ids, attention_mask


def forward_through_graph_encoder(collated):
    global transformerm_model

    inner_states, atom_output = transformerm_model.molecule_encoder(
        collated,
        segment_labels=None,
        perturb=None,
        last_state_only=True
    )

    last_state = inner_states[0]
    molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
    return molecule_embedding


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def data_to_graph(data):
    new_graph = AttrDict()
    new_graph.update(data.to_dict())
    new_graph = preprocess_item(new_graph)
    return new_graph


def smiles_to_transformerm_rep(mol_smiles):
    return graph_to_transformerm_rep(mol_to_graph(Chem.MolFromSmiles(mol_smiles)))


def mol_to_graph(mol):
    graph_as_data = mol_to_graph_data_obj_simple(mol)
    graph = data_to_graph(graph_as_data)
    graph.idx = 0
    graph.y = np.array([0.0])

    return graph


def graph_to_transformerm_rep(graph):
    global device

    max_node = 512
    multi_hop_max_dist = 5
    spatial_pos_max = 1024

    for idx, val in graph.items():
        if isinstance(val, np.ndarray):
            graph[idx] = torch.from_numpy(val)

    collated_graph = collator_3d([graph], max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                                 spatial_pos_max=spatial_pos_max)

    for idx, val in collated_graph.items():
        if hasattr(val, 'to'):
            collated_graph[idx] = val.to(device)

    with torch.no_grad():
        return forward_through_graph_encoder(collated_graph)


# Initialize projection model
import torch
from torch import nn, optim
import pytorch_lightning as pl


class ProjectionModel(pl.LightningModule):
    def __init__(self):
        super(ProjectionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6156, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 768)
        )

        # 6156*4096 + 4096&2048 + 2048*768
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        moflow_rep, transformerm_rep = batch
        output = self(moflow_rep)
        loss = self.criterion(output, transformerm_rep)
        # print(f'{output} vs {transformerm_rep}')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
            }
        }


projection_model = ProjectionModel()
projection_model.load_state_dict(
    torch.load('checkpoints/v3edit-projection-epoch=99-train_loss=0.02.ckpt')['state_dict'], strict=True)
projection_model.to(device)

lr = 0.2
lr_scheduler_steps = 3000
lr_scheduler_gamma = 0.75


def run_z_optimize(model, atomic_num_list, input_text, z, num_steps):
    global text_encoder, device, projection_model, lr, lr_scheduler_steps, lr_scheduler_gamma

    optimizer = torch.optim.Adam([z.requires_grad_()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_scheduler_steps, gamma=lr_scheduler_gamma)
    input_ids, attention_mask = tokenizer_text(input_text, text_max_len=512)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    text_rep = text_encoder(input_ids, attention_mask)
    text_rep = F.normalize(text_rep, dim=-1)

    mse_loss = nn.MSELoss()

    run = [0]
    z_last = None
    skipped_in_a_row = 0
    while run[0] <= num_steps and skipped_in_a_row < 10:
        # print(f'Run: {run[0]}')
        try:
            optimizer.zero_grad()

            adj, atoms = generate_mols_fix(model, atomic_num_list, z_mu=z, batch_size=125,
                                           true_adj=None, temp=1.0)
            z_last = z.clone().detach()

            # x0.backward()
            # print(f'D: z.grad after x0.backward() {z.grad}')
            # print(f'x0: {x0}')
            # xs = x0.softmax(dim=1).long()

            # try:
            # graph = mol_to_graph(mol)
            # graph.x = xs.to('cpu')
            # print(f'xs: {xs}')
            # graph.edge_attr = edge_attr.to('cpu')
            # graph_rep = graph_to_transformerm_rep(graph)
            adj_normalized = rescale_adj(adj)
            z0, _ = model(adj.to(device), atoms.to(device), adj_normalized.to(device))
            h, adj_h = z0
            # Flatten h and adj_h into 1D tensors
            h_flat = h.view(h.shape[0], -1)
            adj_h_flat = adj_h.view(adj_h.shape[0], -1)

            moflow_graph_rep = torch.cat([h_flat, adj_h_flat], dim=1)  # Tensor (batch_size, 6168) long
            # moflow_graph_rep[torch.isnan(moflow_graph_rep)] = 0

            # print(f'moflow_rep: {moflow_graph_rep}')
            graph_rep = projection_model(moflow_graph_rep)  # Transformer-M graph rep. 768

            # except Exception as e:
            #     print(f'Cannot encode the graph!!!: {e}')
            #     print(traceback.format_exc())
            #     break

            graph_rep = F.normalize(graph_rep, dim=-1)
            prompt_loss = -torch.sum(graph_rep @ text_rep.t() / 0.1)
            # original_mol_loss = mse_loss(graph_rep, original_mol_rep)
            loss = torch.nan_to_num(prompt_loss) # + alpha * torch.nan_to_num(original_mol_loss)
            # loss = original_mol_loss
            # print(f'graph_rep: {graph_rep}')

            if torch.isnan(prompt_loss):
                print(loss)
                break

            # print(f'RUN: {run[0]}')
            loss.backward(retain_graph=True)
            # from torchviz import make_dot
            # make_dot(loss).render("loss", format="svg")
            # print(f'z before opt step: {z}')
            # print(f'z grad: {z.grad}')
            optimizer.step()
            # print(f'z after opt step: {z}')
            scheduler.step()
            skipped_in_a_row = 0
        except Exception as e:
            print(f'Skipped run due to: {e}')
            print(traceback.format_exc())
            skipped_in_a_row += 1

        run[0] += 1
        if run[0] % 100 == 0:
            print(f'Optimization Step {run} Loss ({loss:.4f}) = prompt ({prompt_loss:.4f})')

    return z_last, loss.item()


# Generate molecule
# input_smiles = "CCC(=O)Nc1cccc(-n2cnnc2)c1"
# input_text = "This molecule is soluble in water."

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

mu = 0
temperature = 0.05
# alpha = 2000
total_steps = 14000


# Generation
def generate(input_text, attempts=10) -> List[str]:
    global mu, temperature, total_steps

    smiles_to_return = []
    try:
        z = np.zeros(6156)
        z_initials = []
        for i in range(attempts):
            z_augment = np.random.normal(np.ones(6156) * mu, np.ones(6156) * temperature, (1, 6156))
            z_augmented = torch.from_numpy(z + z_augment).float().to(device)
            z_initials.append(z_augmented)

        for z_initial in tqdm(z_initials):
            try:
                z = z_initial
                z.requires_grad = True
                z, loss = run_z_optimize(moflow_model, zinc250_atomic_num_list, input_text, z, total_steps)
                adj, x = generate_mols_fix(moflow_model, zinc250_atomic_num_list, batch_size=100, z_mu=z,
                                           true_adj=None, temp=temperature)
                val_res = check_validity(adj, x, zinc250_atomic_num_list, correct_validity=args.correct_validity)
                valid_smiles = val_res['valid_smiles']
                if valid_smiles and '*' not in valid_smiles[0]:
                    smiles_to_return.append(valid_smiles[0].strip())
            except Exception as e:
                print(f'Generation of molecule {i + 1} failed due to {e}')
                print(traceback.format_exc())
    except Exception as e:
        print(f'Failed to generate molecules: {e}')
        return []

    if not smiles_to_return:
        return []

    return smiles_to_return


def calculate_property(mol, property_name):
    if property_name == 'solubility':
        return Descriptors.MolLogP(mol)
    elif property_name == 'likeness':
        return QED.qed(mol)
    elif property_name == 'permeability':
        return Descriptors.TPSA(mol)
    elif property_name == 'hydrogen_acceptor':
        return Chem.rdMolDescriptors.CalcNumHBA(mol)
    elif property_name == 'hydrogen_donor':
        return Chem.rdMolDescriptors.CalcNumHBD(mol)
    else:
        raise ValueError(f"Unsupported property: {property_name}")


if __name__ == "__main__":
    text = "The molecule contains hydroxyl and carboxyl groups, which can be thermally decomposed to generate ammonia gas, and the oxygen content in the molecule is not less than 20%."
    print(generate(text))
    # Parse arguments
    # new_parser = argparse.ArgumentParser()
    # new_parser.add_argument("--input_file", type=str, default='MoleculeSTM_editing_SMILES.txt')
    # new_parser.add_argument("--output_folder", type=str, default='output')
    # new_parser.add_argument("--molecule_count", type=int, default=None,
    #                         help='Limit amount of molecules to process from the '
    #                              'input file')
    # new_parser.add_argument('--temperature', type=float, default=0.05)
    # new_parser.add_argument('--alpha', type=float, default=2000)
    # new_parser.add_argument('--lr', type=float, default=0.02)
    # new_parser.add_argument('--lr_scheduler_steps', type=int, default=450)
    # new_parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1)
    # new_parser.add_argument('--total_steps', type=int, default=700)
    # new_parser.add_argument('--attempts', type=int, default=2)
    # new_parser.add_argument('--input_text', type=str)
    # new_parser.add_argument('--property', type=str, choices=['solubility', 'likeness', 'permeability', 'hydrogen_acceptor', 'hydrogen_donor'])
    # new_parser.add_argument('--goal', type=str, choices=['increase', 'decrease'])
    #
    # new_args = new_parser.parse_args()
    #
    # # Hyperparameters
    # temperature = new_args.temperature
    # alpha = new_args.alpha
    # lr = new_args.lr
    # lr_scheduler_steps = new_args.lr_scheduler_steps
    # lr_scheduler_gamma = new_args.lr_scheduler_gamma
    # total_steps = new_args.total_steps
    #
    # # Set proper direction for property numerically
    # true_goal = new_args.property
    # if new_args.property in ['solubility', 'permeability']:
    #     # Lower is better for these properties
    #     if new_args.goal == 'increase':
    #         true_goal = 'decrease'
    #     else:
    #         true_goal = 'increase'
    #
    # # Read input file
    # with open(new_args.input_file, 'r') as file:
    #     smiles = [smile.strip() for smile in file.readlines()]
    #
    #     # Limit molecules if needed
    #     if new_args.molecule_count:
    #         smiles = smiles[:new_args.molecule_count]
    #
    # print(f'Loaded {len(smiles)} molecules to process')
    #
    # output_rows = []
    # for idx, smile in enumerate(tqdm(smiles)):
    #     generated_smiles = generate(smile, new_args.input_text, new_args.attempts)
    #
    #     best_smile = None
    #     best_value = None
    #     original_value = None
    #     for generated_smile in generated_smiles:
    #         try:
    #             generated_mol = Chem.MolFromSmiles(generated_smile)
    #             property_score = calculate_property(generated_mol, new_args.property)
    #
    #             if best_value is None or (true_goal == 'increase' and property_score > best_value) or\
    #                     (true_goal == 'decrease' and property_score < best_value):
    #                 best_value = property_score
    #                 best_smile = generated_smile
    #
    #             if original_value is None:
    #                 original_value = calculate_property(Chem.MolFromSmiles(smile), new_args.property)
    #         except Exception as e:
    #             print(f'Failed evaluation for generated smiles {generated_smile} from {smile}: {e}')
    #
    #     delta = best_value - original_value if (best_value and original_value) else None
    #
    #     row = {'id': idx, 'original': smile, 'generated': best_smile, 'delta': delta}
    #     output_rows.append(row)
    #     print(row)
    #
    # if not os.path.exists(new_args.output_folder):
    #     os.makedirs(new_args.output_folder)
    #
    # now = datetime.now()
    # formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    #
    # with open(f'{new_args.output_folder}/args-{formatted_time}.txt', 'w') as file:
    #     file.write(str(new_args))
    #
    # with open(f'{new_args.output_folder}/output-{new_args.property}-{new_args.goal}-{formatted_time}.csv', 'w') as file:
    #     csv_writer = csv.DictWriter(file, fieldnames=['id', 'original', 'generated', 'delta'])
    #     csv_writer.writeheader()
    #     csv_writer.writerows(output_rows)



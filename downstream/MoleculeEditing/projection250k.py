#!/usr/bin/env python
# coding: utf-8

# In[1]:

import faulthandler
import random

from pytorch_lightning.strategies import DDPStrategy, DDPSpawnStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau

faulthandler.enable()

import csv

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

smiles = []
with open('zinc250k.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        smiles.append(row['smiles'])


# In[2]:


with open('MoleculeSTM_editing_SMILES.txt', 'r') as file:
    smiles.extend(file.readlines())


# In[3]:


len(smiles)


# In[4]:


import sys
import importlib

sys.path.insert(0, '../graph-transformer')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d


# In[5]:


import os
import time
import argparse
from distutils.util import strtobool
from mflow.models.hyperparams import Hyperparameters
from mflow.utils.model_utils import load_model, get_latent_vec
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, default='This molecule is beautiful.')
parser.add_argument("--checkpoint_name", type=str, default='littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt')
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
args = parser.parse_args("--model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask -snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k --hyperparams-path moflow-params.json --temperature 0.85 --batch-size 1 --n_experiments 5".split(" "))

start = time.time()
print("Start at Time: {}".format(time.ctime()))
# chainer.config.train = False
snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
print("loading hyperparamaters from {}".format(hyperparams_path))
model_params = Hyperparameters(path=hyperparams_path)

device = None
num_gpus = torch.cuda.device_count()
# Random GPU
if num_gpus > 0:
    # Select a random GPU
    device_id = random.randint(0, num_gpus - 1)
    device = f'cuda:{device_id}'
else:
    device = 'cpu'

model = None


# In[6]:


from mflow.utils.model_utils import smiles_to_adj, rescale_adj


def smiles_to_moflow_rep(smiles):
    global model, device

    if model is None:
        model = load_model(snapshot_path, model_params, debug=True)
        if len(model.ln_var) == 1:
            print('model.ln_var: {:.2f}'.format(model.ln_var.item()))
        elif len(model.ln_var) == 2:
            print('model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}'.format(model.ln_var[0].item(),
                                                                            model.ln_var[1].item()))

        model = model.to(device)
        print(f'device = {device}')
        model.eval()  # Set model for evaluation

    adj, atoms = smiles_to_adj(smiles, 'zinc250k')
    adj_normalized = rescale_adj(adj)
    with torch.no_grad():
        z0, _ = model(adj.to(device), atoms.to(device), adj_normalized.to(device))

    h, adj_h = z0
    # Flatten h and adj_h into 1D tensors
    h_flat = h.view(h.shape[0], -1)
    adj_h_flat = adj_h.view(adj_h.shape[0], -1)

    return torch.cat([h_flat, adj_h_flat], dim=1)


# In[7]:


from torch import Tensor
from types import SimpleNamespace
from torch_geometric.data import Data
import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit.Chem import AllChem
from rdkit import Chem


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
    if AllChem.EmbedMolecule(mol) == -1:
        bad = True
    # AllChem.EmbedMolecule(mol)

    mol_try = Chem.Mol(mol)
    if not bad:
        try:
            AllChem.MMFFOptimizeMolecule(mol_try)
            mol = mol_try
        except ValueError:
            print("oops")
            pass

    mol = Chem.RemoveHs(mol)

    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        # atom_feature = [allowable_features['possible_atomic_num_list'].index(
        #     atom.GetAtomicNum())] + [allowable_features[
        #     'possible_chirality_list'].index(atom.GetChiralTag())]
        # atom_features_list.append(atom_feature)
        atom_features_list.append(atom_to_feature_vector(atom))

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
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
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    # positions
    # try:
    #     if not bad:
    #         positions = mol.GetConformer().GetPositions()
    #     else:
    #         num_atoms = mol.GetNumAtoms()
    #         positions = np.zeros((num_atoms, 3))
    # except:
    #     num_atoms = mol.GetNumAtoms()
    #     positions = np.zeros((num_atoms, 3))
    num_atoms = mol.GetNumAtoms()
    positions = np.zeros((num_atoms, 3))

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

    data = Data(x=x, edge_index=torch.from_numpy(edge_index).to(torch.int64), edge_attr=torch.from_numpy(edge_attr).to(torch.int64))

    data.__num_nodes__ = len(x)
    data.pos = positions

    return data

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def data_to_graph(data):
    new_graph = AttrDict()
    new_graph.update(data.to_dict())
    new_graph = preprocess_item(new_graph)
    return new_graph

args = SimpleNamespace(_name='transformer_m_base', act_dropout=0.3, activation_fn='gelu', add_3d=True,
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

transformer = None


def forward_through_graph_encoder(collated):
    global transformer, args, device

    if transformer is None:
        transformer = TransformerM(args=args)
        transformer = transformer.to(device)

        d = torch.load(
            '../model-epoch=507.ckpt')
        d = d['state_dict']
        state_dict = {}
        for k, v in d.items():
            if k.startswith('graph_encoder.'):
                state_dict[k.removeprefix('graph_encoder.')] = v

        missing_keys, unexpected_keys = transformer.load_state_dict(state_dict)
        del state_dict
        print('Missing Keys: ' + str(missing_keys))
        print('Unexpected Keys: ' + str(unexpected_keys))

        transformer.eval()

    inner_states, atom_output = transformer.molecule_encoder(
        collated,
        segment_labels=None,
        perturb=None,
        last_state_only=True
    )

    last_state = inner_states[0]
    molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
    return molecule_embedding


def smiles_to_transformerm_rep(mol_smiles):
    global transformer, device
    data = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(mol_smiles))
    graph = data_to_graph(data)
    graph.idx = 0

    graph.y = np.array([0.0])

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

    # transformer = transformer.to('cuda:0')
    with torch.no_grad():
        return forward_through_graph_encoder(collated_graph)


# In[8]:


m_rep = smiles_to_moflow_rep("CCCC")
m_rep, m_rep.shape


# In[9]:


tm_rep = smiles_to_transformerm_rep("CCCC")
tm_rep, tm_rep.shape

# In[10]:

from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, smiles, cache_dir):
        self.smiles = smiles
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        cache_file = os.path.join(self.cache_dir, f"{idx}.pt")
        if os.path.exists(cache_file):
            moflow_rep, transformerm_rep = torch.load(cache_file)
        else:
            mol_smiles = self.smiles[idx]
            moflow_rep = smiles_to_moflow_rep(mol_smiles).squeeze(0).to('cpu')
            transformerm_rep = smiles_to_transformerm_rep(mol_smiles).squeeze(0).to('cpu')
            torch.save((moflow_rep, transformerm_rep), cache_file)
        return moflow_rep, transformerm_rep


dataset = EmbeddingsDataset(smiles, 'embeddings_cache/')


# In[11]:


d = dataset[0]
d[0].shape, d[1].shape


# In[16]:


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

        #6156*4096 + 4096&2048 + 2048*768
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


# In[18]:
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

# In[19]:

if __name__ == "__main__":
    project_model = ProjectionModel()

    # Check if multiple GPUs are available and use all of them
    gpus = -1 if torch.cuda.device_count() > 1 else None

    # Define the WandbLogger
    wandb_logger = WandbLogger(project="Molecule Edit Alignment")

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',  # path where to save model checkpoints
        filename='v3edit-projection-{epoch:02d}-{train_loss:.2f}',  # filename pattern
        save_top_k=-1,  # save model after every epoch
        verbose=True,  # give some logs
        monitor='train_loss',  # logged metric to monitor
        mode='min'  # 'min' for metrics where lower is better
    )

    # Get the latest checkpoint
    checkpoints = [f for f in os.listdir('checkpoints/') if 'v3' in f]
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1] if checkpoints else None

    trainer = pl.Trainer(max_epochs=500, strategy=DDPSpawnStrategy(find_unused_parameters=False), gpus=gpus,
                         logger=wandb_logger, callbacks=[checkpoint_callback],
                         precision=16, log_every_n_steps=1,
                         resume_from_checkpoint='checkpoints/' + latest_checkpoint if latest_checkpoint else None)

    trainer.fit(project_model, train_loader)


# In[ ]:





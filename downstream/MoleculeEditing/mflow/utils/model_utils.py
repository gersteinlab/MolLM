import torch
import numpy as np
from data.smile_to_graph import GGNNPreprocessor
from rdkit import Chem

from data import transform_qm9
from data.transform_zinc250k import one_hot_zinc250k, transform_fn_zinc250k
from mflow.models.model import MoFlow as Model


def load_model(snapshot_path, model_params, debug=False):
    print("loading snapshot: {}".format(snapshot_path))
    if debug:
        print("Hyper-parameters:")
        model_params.print()
    model = Model(model_params)

    device = torch.device('cpu')
    model.load_state_dict(torch.load(snapshot_path, map_location=device))
    return model


def smiles_to_adj(mol_smiles, data_name='qm9'):
    out_size = 9
    transform_fn = transform_qm9.transform_fn

    if data_name == 'zinc250k':
        out_size = 38
        transform_fn = transform_fn_zinc250k

    preprocessor = GGNNPreprocessor(out_size=out_size, kekulize=True)
    canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(Chem.MolFromSmiles(mol_smiles)) # newly added crucial important!!!
    atoms, adj = preprocessor.get_input_features(mol)
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)

    adj = torch.from_numpy(adj)
    atoms = torch.from_numpy(atoms)
    return adj, atoms


def rescale_adj(adj, type='all'):
    # Previous paper didn't use rescale_adj.
    # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
    # In this implementation, the normaliztion term is different
    # raise NotImplementedError
    # (256,4,9, 9):
    # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
    # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
    # usually first 3 matrices have no diagnal, the last has.
    # A_prime = self.A + sp.eye(self.A.shape[0])
    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime


def get_latent_vec(model, mol_smiles, data_name='qm9'):
    adj, atoms = smiles_to_adj(mol_smiles, data_name)
    adj_normalized = rescale_adj(adj)
    with torch.no_grad():
        device = next(model.parameters()).device
        z = model(adj.to(device), atoms.to(device), adj_normalized.to(device))
    z = np.hstack([z[0][0].cpu().numpy(), z[0][1].cpu().numpy()]).squeeze(0) # change later !!! according to debug
    return z


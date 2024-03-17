import importlib
import sys

import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from transformers import BertTokenizer

def graph_to_transformer_rep(graph, transformer_model):
    max_node = 512
    multi_hop_max_dist = 5
    spatial_pos_max = 1024

    for idx, val in graph.items():
        if isinstance(val, np.ndarray):
            graph[idx] = torch.from_numpy(val)
            if idx == 'y':
                graph['y'] = graph['y'].to(torch.float32)
            elif idx == 'pos':
                graph['pos'] = graph['pos'].to(torch.float32)

    collated_graph = collator_3d([graph], max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                                 spatial_pos_max=spatial_pos_max)

    for idx, val in collated_graph.items():
        if hasattr(val, 'to'):
            collated_graph[idx] = val.to(next(transformer_model.parameters()).device)

    with torch.no_grad():
        return forward_through_graph_encoder(collated_graph, transformer_model)


def forward_through_graph_encoder(collated, transformer_model):
    collated = collated
    inner_states, atom_output = transformer_model.molecule_encoder(
        collated,
        segment_labels=None,
        perturb=None,
        last_state_only=True
    )

    last_state = inner_states[0]
    molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
    return molecule_embedding


def smiles_to_transformer_rep(mol_smiles, transformer_model):
    return graph_to_transformer_rep(mol_to_graph(Chem.MolFromSmiles(mol_smiles)), transformer_model)


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

    data = Data(x=x, edge_index=torch.from_numpy(edge_index).to(torch.int64),
                edge_attr=torch.from_numpy(edge_attr).to(torch.int64))

    data.__num_nodes__ = len(x)
    data.pos = positions

    return data


def mol_to_graph(mol):
    graph_as_data = mol_to_graph_data_obj_simple(mol)
    graph = data_to_graph(graph_as_data)
    graph.idx = 0
    graph.y = np.array([0.0])

    return graph


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def data_to_graph(data):
    new_graph = AttrDict()
    new_graph.update(data.to_dict())
    new_graph = preprocess_item(new_graph)
    return new_graph


sys.path.insert(0, './')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d


class MolLM(torch.nn.Module):
    def __init__(self, molecule_checkpoint_path, model_code_dir, tokenizer_path):
        super(MolLM, self).__init__()
        sys.path.insert(0, model_code_dir)
        GraphTransformer = importlib.import_module("model.contrastive_gin")
        self.molecule_encoder = GraphTransformer.GINSimclr.load_from_checkpoint(molecule_checkpoint_path, strict=False)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    def forward_molecule(self, smiles):
        return smiles_to_transformer_rep(smiles, self.molecule_encoder.graph_encoder)

    def forward_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=512,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        with torch.no_grad():
            return self.molecule_encoder.text_encoder(input_ids, attention_mask)

import gc
import io
import random
import re
import traceback
from typing import List, Tuple
from zipfile import ZipFile

import numpy as np
import torch
from auglichem.molecule import MotifRemoval
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from lock import *
from task import Task


class AugTask(Task[Tuple[str, int]]):

    def get_batch(self, batch_size: int) -> List[Tuple[str, int]]:
        try:
            select_sql = """SELECT zip_path, epoch
                                FROM v2_aug_batches 
                                WHERE in_progress = 0
                                ORDER BY epoch, zip_path
                                LIMIT %s
                                FOR UPDATE"""
            results = DB.fetch_all(select_sql, (batch_size,), commit=False)

            if results is None:
                return []

            tuples = [(row['zip_path'], row['epoch']) for row in results]

            # Mark as in_process
            update_sql = """UPDATE v2_aug_batches
                                SET in_progress = 1
                                WHERE zip_path = %s AND epoch=%s"""
            for zip_path, epoch in tuples:
                DB.execute(update_sql, (zip_path, epoch), commit=False)
            DB.commit()
        except Exception as e:
            print("An error occurred: ", e)
            print(traceback.format_exc())
            DB.rollback()
            return []

        DB.disconnect()
        return tuples

    def process(self, batch):
        DB.disconnect()
        zip_path, epoch = batch

        RDLogger.DisableLog('rdApp.*')
        # Scan file
        cid_to_smiles = {}

        # processed = 0
        try:
            acquire_lock(zip_path)
            with ZipFile(zip_path, 'a') as zip_file:
                current_cids = []

                existing_files = zip_file.namelist()
                for file_name in existing_files:
                    match = re.search(r"mol_(\d+)_smiles\.txt", file_name)
                    if match:
                        cid = match.group(1)
                        if f'mol_{cid}_text.txt' in existing_files:
                            current_cids.append(cid)

                for cid in current_cids:
                    try:
                        with zip_file.open(f'mol_{cid}_smiles.txt') as smiles_file:
                            # Read the file content into a string
                            smiles = smiles_file.read().decode('utf-8')
                            cid_to_smiles[int(cid)] = smiles
                    except:
                        pass
        finally:
            release_lock(zip_path)

        # Remember buffers to save to file
        zip_internal_path_to_buffer = {}

        # Process CIDs
        for cid, smiles in cid_to_smiles.items():
            # print(f'Processing {processed}: {smiles}')
            try:
                mol = Chem.MolFromSmiles(smiles)

                # Augment
                augment_result = augment_molecule(mol)

                # Original
                orig_pt_path = f'mol_{cid}_original.pt'
                if orig_pt_path not in existing_files:
                    zip_internal_path_to_buffer[orig_pt_path] = io.BytesIO()
                    torch.save(mol_to_graph_data_obj_simple(mol), zip_internal_path_to_buffer[orig_pt_path])

                # Save Augments
                keys = ['aug_1', 'aug_2', 'aug_3', 'aug_4']
                for key in keys:
                    if augment_result[key] is None:
                        continue
                    aug_path = f'mol_{cid}_e{epoch}_{key}.pt'
                    if aug_path not in existing_files:
                        zip_internal_path_to_buffer[aug_path] = io.BytesIO()
                        torch.save(mol_to_graph_data_obj_simple(augment_result[key]),
                                   zip_internal_path_to_buffer[aug_path])

                # processed += 1
                # print(f'In-memory processed {processed} / {len(cid_to_smiles)}.')
            except Exception as e:
                print(traceback.format_exc())
                print(f'Failed to process {cid} in {zip_path}: {e}')

        # Save buffers to zip file
        try:
            acquire_lock(zip_path)
            print(f'Writing to zip {zip_path}')
            with ZipFile(zip_path, 'a') as zip_file:
                existing_files = zip_file.namelist()
                for path, buffer in zip_internal_path_to_buffer.items():
                    if path in existing_files:
                        continue

                    zip_file.writestr(path, buffer.getvalue())
        finally:
            release_lock(zip_path)

        gc.collect()

    def mark_complete(self, batch):
        zip_path, epoch = batch
        update_sql = """UPDATE v2_aug_batches
                                    SET finished=1
                                    WHERE zip_path = %s AND epoch=%s"""
        DB.execute(update_sql, (zip_path, epoch))
        DB.disconnect()


def get_path_indices(num):
    # Break down the input number into two separate numbers
    num_str = str(num)
    while len(num_str) < 9:
        num_str = "0" + num_str
    part1 = num_str[:3]
    part2 = num_str[3:6]

    # Convert each part into a string and pad with leading zeroes
    part1 = part1.zfill(3)
    part2 = part2.zfill(3)
    return part1, part2


def get_folder_and_zip_path(indices):
    part1, part2 = indices
    folder = f"output-text/{part1}/"
    tar = folder + f"{part2}.zip"
    return folder, tar

# JEFF
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


def save_augments(files, mol_smiles):
    assert (len(files) == 5)
    mol = Chem.MolFromSmiles(mol_smiles)
    augs = list(augment_molecule(mol).values())
    for aug in augs:
        print(aug)
    for i, f in enumerate(files):
        if augs[i] is not None:
            augs[i] = mol_to_graph_data_obj_simple(augs[i])
        torch.save(augs[i], f)


def augment_molecule(mol):  # returns dict of mols and augments
    RDLogger.DisableLog('rdApp.*')

    mol = duplicate(mol)
    mol = Chem.RemoveHs(mol)

    mol3d = duplicate(mol)
    mol3d = Chem.AddHs(mol3d)
    try:
        AllChem.EmbedMolecule(mol3d)
        AllChem.MMFFOptimizeMolecule(mol3d)
    except:
        print(f'Fail for {mol}')

    mol3d = Chem.RemoveHs(mol3d)

    result = {'original': duplicate(mol),
              'aug_1': augment_1_drop(duplicate(mol)),
              'aug_2': augment_2_walk(duplicate(mol)),
              'aug_3': augment_3_chem_react(duplicate(mol), mol3d),
              'aug_4': augment_4_motif_removal(duplicate(mol))}

    return result


def add_3d(mol):
    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol)
        mol_try = Chem.Mol(mol)
        AllChem.MMFFOptimizeMolecule(mol_try)
        mol = mol_try
    except:
        return mol

    mol = Chem.RemoveHs(mol)
    return mol


def augment_1_drop(mol, p=0.10):
    tries = 0
    mol_copy = None
    while mol_copy == None and tries < 5:
        tries += 1
        try:
            emol = Chem.EditableMol(mol)
            num_atoms_to_remove = int(mol.GetNumAtoms() * p)
            if num_atoms_to_remove < 1:
                num_atoms_to_remove = 1
            # num_atoms_to_remove = 0

            # Remove atoms
            all_idxs = list(range(mol.GetNumAtoms()))
            atoms_to_remove = random.sample(all_idxs, num_atoms_to_remove)
            remaining_original_ids = [idx for idx in all_idxs if idx not in atoms_to_remove]

            atom_map = []
            for new_id, original_id in enumerate(remaining_original_ids):
                atom_map.append((new_id, original_id))

            for atom_idx in sorted(atoms_to_remove, reverse=True):
                emol.RemoveAtom(atom_idx)
            mol_copy = emol.GetMol()

            # Optimize the molecule with the remaining atoms using the MMFF force field

            Chem.SanitizeMol(mol_copy)
            mol_copy = Chem.AddHs(mol_copy)
            AllChem.EmbedMolecule(mol_copy)
            AllChem.MMFFOptimizeMolecule(mol_copy)
            mol_copy = Chem.RemoveHs(mol_copy)

            Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
        except Exception as e:
            print(e)
            mol_copy = None
    return mol_copy


def augment_2_walk(mol):
    tries = 0
    mol_copy = None
    while mol_copy == None and tries < 5:
        tries += 1
        try:
            # Get the number of atoms in the molecule
            num_atoms = mol.GetNumAtoms()

            # Calculate the desired walk length based on the coverage
            walk_length = int(0.8 * num_atoms)

            # Initialize the walk with a random starting atom
            walk = [random.randint(0, num_atoms - 1)]

            # Perform the random walk
            for _ in range(walk_length - 1):
                # Get the last atom in the walk
                current_atom = mol.GetAtomWithIdx(walk[-1])

                # Get neighboring atoms
                neighbors = [a.GetIdx() for a in current_atom.GetNeighbors()]

                # Choose a random neighbor that is not already in the walk, if possible
                available_neighbors = list(set(neighbors) - set(walk))
                if available_neighbors:
                    next_atom = random.choice(available_neighbors)
                else:
                    # If all neighbors are already in the walk, choose a random neighbor
                    next_atom = random.choice(neighbors)

                walk.append(next_atom)

            # Create the subgraph by extracting atoms and bonds from the walk
            atom_indices = set(walk)

            all_idxs = list(range(mol.GetNumAtoms()))
            atoms_to_remove = [idx for idx in all_idxs if idx not in atom_indices]
            remaining_original_ids = [idx for idx in all_idxs if idx not in atoms_to_remove]

            emol = Chem.EditableMol(mol)
            atom_map = []
            for new_id, original_id in enumerate(remaining_original_ids):
                atom_map.append((new_id, original_id))

            for atom_idx in sorted(atoms_to_remove, reverse=True):
                emol.RemoveAtom(atom_idx)
            mol_copy = emol.GetMol()

            # Optimize the molecule with the remaining atoms using the MMFF force field
            Chem.SanitizeMol(mol_copy)
            mol_copy = Chem.AddHs(mol_copy)
            AllChem.EmbedMolecule(mol_copy)
            AllChem.MMFFOptimizeMolecule(mol_copy)
            mol_copy = Chem.RemoveHs(mol_copy)

            Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)

        except Exception as e:
            mol_copy = None
    return mol_copy


def augment_3_methylate(mol, reference_mol):
    try:
        hydrogen_candidates = [atom.GetIdx() for atom in mol.GetAtoms() if (atom.GetNumImplicitHs() > 0)]
        if not hydrogen_candidates:
            return "no candidates"

        num_atoms = mol.GetNumAtoms()

        # Choose a random atom index from the candidates
        random_atom_idx = random.choice(hydrogen_candidates)

        mapping = []
        for i in range(num_atoms):
            mapping.append((i, i))

        # Get the editable version of the molecule
        rw_mol = Chem.RWMol(mol)

        # Add a methyl group to the chosen atom
        new_atom = Chem.Atom(6)  # Carbon atom for the methyl group
        rw_mol.AddAtom(new_atom)
        methyl_idx = rw_mol.GetNumAtoms() - 1
        rw_mol.AddBond(random_atom_idx, methyl_idx, Chem.BondType.SINGLE)

        mol_copy = rw_mol.GetMol()

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
        mol_copy = Chem.AddHs(mol_copy)
        AllChem.EmbedMolecule(mol_copy)
        AllChem.MMFFOptimizeMolecule(mol_copy)
        mol_copy = Chem.RemoveHs(mol_copy)

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
    except:
        mol_copy = None
    return mol_copy


def augment_3_demethylate(mol, reference_mol):
    try:
        # Find all methyl groups
        methyl_indices = [atom.GetIdx() for atom in mol.GetAtoms() if
                          atom.GetSymbol() == "C" and atom.GetNumImplicitHs() == 3]

        # Return None if no methyl groups are found
        if not methyl_indices:
            return "no candidates"

        # Choose a random methyl group
        methyl_idx = random.choice(methyl_indices)

        num_atoms = mol.GetNumAtoms()

        # Get the neighboring atom (connected atom) of the methyl group
        methyl = mol.GetAtomWithIdx(methyl_idx)
        connected_atom_idx = [neighbor.GetIdx() for neighbor in methyl.GetNeighbors()][0]

        # Create an editable molecule
        emol = Chem.EditableMol(mol)

        # Remove the methyl group
        emol.RemoveAtom(methyl_idx)
        mapping = []
        old_ids = list(range(num_atoms))
        new_id = 0
        for old_id in old_ids:
            if old_id == methyl_idx:
                continue
            mapping.append((new_id, old_id))
            new_id += 1

        # Convert the editable molecule back to a regular molecule
        updated_mol = emol.GetMol()

        # Update the connected_atom_idx if necessary
        if connected_atom_idx > methyl_idx:
            connected_atom_idx -= 1

        # Create a new editable molecule from the updated molecule
        emol = Chem.EditableMol(updated_mol)

        # Add a hydrogen atom
        hydrogen_idx = emol.AddAtom(Chem.Atom("H"))
        emol.AddBond(connected_atom_idx, hydrogen_idx, Chem.BondType.SINGLE)

        # Convert the editable molecule back to a regular molecule
        mol_copy = emol.GetMol()
        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
        mol_copy = Chem.AddHs(mol_copy)
        AllChem.EmbedMolecule(mol_copy)
        AllChem.MMFFOptimizeMolecule(mol_copy)
        mol_copy = Chem.RemoveHs(mol_copy)

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
    except:
        mol_copy = None
    return mol_copy


def augment_3_aminate(mol, reference_mol):
    try:
        hydrogen_candidates = [atom.GetIdx() for atom in mol.GetAtoms() if (atom.GetNumImplicitHs() > 0)]
        if not hydrogen_candidates:
            return "no candidates"

        num_atoms = mol.GetNumAtoms()

        # Choose a random atom index from the candidates
        random_atom_idx = random.choice(hydrogen_candidates)

        mapping = []
        for i in range(num_atoms):
            mapping.append((i, i))

        # Get the editable version of the molecule
        rw_mol = Chem.RWMol(mol)

        # Add a methyl group to the chosen atom
        new_atom = Chem.Atom(7)  # Nitrogen atom for the amine group
        rw_mol.AddAtom(new_atom)
        amino_idx = rw_mol.GetNumAtoms() - 1
        rw_mol.AddBond(random_atom_idx, amino_idx, Chem.BondType.SINGLE)

        mol_copy = rw_mol.GetMol()

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
        mol_copy = Chem.AddHs(mol_copy)
        AllChem.EmbedMolecule(mol_copy)
        AllChem.MMFFOptimizeMolecule(mol_copy)
        mol_copy = Chem.RemoveHs(mol_copy)

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
    except:
        mol_copy = None
    return mol_copy


def augment_3_deaminate(mol, reference_mol):
    try:
        # Find all methyl groups
        amino_candidates = [atom.GetIdx() for atom in mol.GetAtoms() if
                            atom.GetSymbol() == "N" and atom.GetNumImplicitHs() >= 2]

        # Return None if no methyl groups are found
        if not amino_candidates:
            return "no candidates"

        # Choose a random methyl group
        amino_idx = random.choice(amino_candidates)

        num_atoms = mol.GetNumAtoms()

        # Get the neighboring atom (connected atom) of the methyl group
        methyl = mol.GetAtomWithIdx(amino_idx)
        connected_atom_idx = [neighbor.GetIdx() for neighbor in methyl.GetNeighbors()][0]

        # Create an editable molecule
        emol = Chem.EditableMol(mol)

        # Remove the methyl group
        emol.RemoveAtom(amino_idx)
        mapping = []
        old_ids = list(range(num_atoms))
        new_id = 0
        for old_id in old_ids:
            if old_id == amino_idx:
                continue
            mapping.append((new_id, old_id))
            new_id += 1

        # Convert the editable molecule back to a regular molecule
        updated_mol = emol.GetMol()

        # Update the connected_atom_idx if necessary
        if connected_atom_idx > amino_idx:
            connected_atom_idx -= 1

        # Create a new editable molecule from the updated molecule
        emol = Chem.EditableMol(updated_mol)

        # Add a hydrogen atom
        hydrogen_idx = emol.AddAtom(Chem.Atom("H"))
        emol.AddBond(connected_atom_idx, hydrogen_idx, Chem.BondType.SINGLE)

        # Convert the editable molecule back to a regular molecule
        mol_copy = emol.GetMol()
        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
        mol_copy = Chem.AddHs(mol_copy)
        AllChem.EmbedMolecule(mol_copy)
        AllChem.MMFFOptimizeMolecule(mol_copy)
        mol_copy = Chem.RemoveHs(mol_copy)

        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SANITIZE_ALL)
    except:
        mol_copy = None
    return mol_copy


def augment_3_chem_react(mol, mol3d, p=.1):
    # Attempt to perform num_atoms * p augmentations from augment_3 (min 2)

    n = mol.GetNumAtoms()
    num_reacts = int(n * p)

    funcs = augment_3_functions.copy()
    random.shuffle(funcs)
    augs_3_done = 0

    current_mol = mol
    aug_names = []
    new_mol = None

    if num_reacts < 2:
        num_reacts = 2

    for _ in range(num_reacts):
        for i in range(len(funcs) - 1, -1, -1):
            func = funcs[i]
            # Do only num_reacts amt
            if augs_3_done >= num_reacts or current_mol is None:
                break

            Chem.SanitizeMol(current_mol)

            new_mol = None
            try:
                new_mol = func(current_mol, mol3d)
            except:
                pass

            if new_mol == "no candidates":
                del funcs[i]
            elif new_mol:
                # Good
                aug_names.append(func.__name__)
                augs_3_done += 1

                if augs_3_done == 1:
                    # Remove 3d data from new_mol for next
                    try:
                        current_mol = duplicate(new_mol)
                        current_mol = Chem.RemoveHs(current_mol)
                        smiles = Chem.MolToSmiles(current_mol)
                        current_mol = Chem.MolFromSmiles(smiles)
                    except:
                        pass

    if augs_3_done == 0:
        return None
    return current_mol


def augment_4_motif_removal(mol):
    transform = MotifRemoval(0)
    augs = transform.apply_transform(mol)
    if len(augs) == 0:
        return None
    rand_idx = random.randint(0, len(augs) - 1)
    return augs[rand_idx]


def duplicate(mol):
    return Chem.Mol(mol)


augment_3_functions = [augment_3_methylate, augment_3_demethylate, augment_3_aminate, augment_3_deaminate]
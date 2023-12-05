from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import argparse
import pandas as pd
import json
from datetime import datetime


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


def get_deltas(zipped_mols, property_name):
    deltas = dict()
    for mol in zipped_mols:
        deltas[mol[0]] = calculate_property(mol[2], property_name) - calculate_property(mol[1], property_name)
    
    return deltas


def main():
    parser = argparse.ArgumentParser(description='evaluate editing task')
    parser.add_argument('--file', type=str, default="data.csv")
    parser.add_argument('--metric', type=str, default="solubility")
    parser.add_argument('--delta', type=float, default=0)
    parser.add_argument('--output_path', type=str, default="eval_results/")
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    data = []

    for _,row in df.iterrows():
        data.append((row['id'], Chem.MolFromSmiles(row['original']), Chem.MolFromSmiles(row['generated'])))

    deltas = get_deltas(data, args.metric)

    delta_filter = (lambda x: x >= args.delta) if args.delta >= 0 else (lambda x: x <= args.delta)
    hit_ratio = len(list(filter(delta_filter, deltas.values()))) / len(deltas.values())

    print(f"deltas: {deltas} \n")
    print(f"hit_ratio: {hit_ratio}")

    final_save = {"hit_ratio": hit_ratio, "deltas":deltas}

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M")

    with open(f"{args.output_path}data_{args.metric}_delta={args.delta}_{formatted_now}.json", 'w') as f:
        json.dump(final_save, f)



if __name__ == "__main__":
    main()
#%%
from zipfile import ZipFile

import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer, RobertaTokenizer

cid = "5002"
which_zip = "000/005.zip"
zip_path = f"/gpfs/slayman/pi/gerstein/xt86/ismb2023/Transformer-MoMu/new/dataset-creation/worker/output-text-backup-6-15/{which_zip}"
epoch = 1

with ZipFile(zip_path, 'a') as zip_file:
    files = zip_file.namelist()

    for s in files:
        if "5002" in s:
            print(s)

    text_file_name = f'mol_{cid}_text.txt'
    if text_file_name not in files:
        print("oops")
    else:
        print("not oops")
    with zip_file.open(text_file_name) as text_file:
        texts = text_file.read().decode('UTF-8').split('\n')

    original_pt_name = f'mol_{cid}_original.pt'
    keys = ['aug_1', 'aug_2', 'aug_3', 'aug_4']
    aug_pt_names = [f'mol_{cid}_e{epoch}_{key}.pt' for key in keys]
    has_any_aug = False
    for aug_pt_name in aug_pt_names:
        if aug_pt_name in files:
            has_any_aug = True
            break
    if not has_any_aug:
        print("oops")

    original_pt = torch.load(zip_file.open(original_pt_name))

    augs = []
    for aug_pt_name in aug_pt_names:
        if aug_pt_name in files:
            augs.append(torch.load(zip_file.open(aug_pt_name)))

    texts_to_select = 3
    if len(texts) >= 3:
        selected_texts = random.sample(texts, texts_to_select)
    else:
        selected_texts = []
        for _ in range(3):
            selected_texts.append(random.choice(texts))

    print(selected_texts)

#%%
for i in range(10):
    print(i)
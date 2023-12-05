import gc
import pickle
import re
from typing import Sequence
from zipfile import ZipFile

import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer, RobertaTokenizer


def dict_to_data(data_dict):
    data = Data()

    for key, value in data_dict.items():
        if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            # Handling lists of tensors
            setattr(data, key, [t.clone() for t in value])
        elif isinstance(value, torch.Tensor):
            # Handling tensors
            setattr(data, key, value.clone())
        else:
            # Handling any other attributes (scalars, etc.)
            setattr(data, key, value)
    return data


class GINPretrainDataset(Dataset):
    # Folder format
    # graph_dicts: graph_{NEW ID}.pickle
    # new_text: text_{NEW ID}.txt

    def __init__(self, root, text_max_len, graph_aug1, graph_aug2):
        super(GINPretrainDataset, self).__init__(root)
        # self.root = root
        # self.graph_aug1 = graph_aug1
        # self.graph_aug2 = graph_aug2
        # self.text_max_len = text_max_len
        # # self.graph_folder = root+'../../../data/graph_dicts/'
        # # self.graph_name_list = os.listdir(self.graph_folder)
        #
        # # Find all the ids
        # # # print(self.graph_name_list)
        # # self.valid_ids = [int(re.search(r'\d+', name).group()) for name in self.graph_name_list]
        # # self.valid_ids = sorted(self.valid_ids)
        # #
        #
        # self.text_folder = root + '../../../data/new_text/'
        #
        # self.epoch = None
        # self.aug1_folder = None
        # self.aug2_folder = None
        # self.aug3_folder = None
        # self.valid_ids = []
        self.epoch = None
        self.dataset_folder = './output-text-backup-6-28/'
        self.valid_cids = []
        with open('cids.txt', 'r') as cid_file:
            for line in cid_file:
                self.valid_cids.append(int(line.strip()))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.valid_cids)

    def get(self, index):
        return self.__getitem__(index)

    def indices(self) -> Sequence:
        return range(len(self.valid_ids))

    @staticmethod
    def read_to_data(file_path):
        data_dict = None
        with open(file_path, 'rb') as graph_pickle_file:
            gc.disable()
            data_dict = pickle.load(graph_pickle_file)
        gc.enable()

        if data_dict:
            return dict_to_data(data_dict)
        else:
            return None

    @staticmethod
    def get_random_substring(string, length=128):
        if len(string) <= length:
            return string
        else:
            start = random.randint(0, len(string) - length)
            return string[start:start + length]

    @staticmethod
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

    @staticmethod
    def get_folder_and_zip_path(indices):
        part1, part2 = indices
        folder = f"/gpfs/slayman/pi/gerstein/xt86/ismb2023/Transformer-MoMu/new/dataset-creation/worker/output-text-backup-6-28/{part1}/"
        tar = folder + f"{part2}.zip"
        return folder, tar

    def __getitem__(self, index):
        # # Convert to new_id
        # try:
        #     index = self.valid_ids[index]
        # except:
        #     print(f'Invalid index: {index}')
        #     index = random.choice(self.valid_ids)
        #
        #
        cid = self.valid_cids[index]
        _, zip_path = GINPretrainDataset.get_folder_and_zip_path(GINPretrainDataset.get_path_indices(cid))
        try:
            with ZipFile(zip_path, 'a') as zip_file:
                files = zip_file.namelist()

                text_file_name = f'mol_{cid}_text.txt'
                if text_file_name not in files:
                    return None
                with zip_file.open(text_file_name) as text_file:
                    texts = text_file.read().decode('UTF-8').split('\n')

                original_pt_name = f'mol_{cid}_original.pt'
                keys = ['aug_1', 'aug_2', 'aug_3', 'aug_4']
                aug_pt_names = [f'mol_{cid}_e{self.epoch}_{key}.pt' for key in keys]
                has_any_aug = False
                for aug_pt_name in aug_pt_names:
                    if aug_pt_name in files:
                        has_any_aug = True
                        break
                if not has_any_aug:
                    return None

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

                selected_texts_tokenized = []
                selected_texts_masks = []
                for selected_text in selected_texts:
                    tokens, mask = self.tokenizer_text(selected_text)
                    selected_texts_tokenized.append(tokens.squeeze(0))
                    selected_texts_masks.append(mask.squeeze(0))

                return original_pt, augs, selected_texts_tokenized, selected_texts_masks
        except Exception as e:
            print(f'Failed to process {zip_path}: {e}')
            return None
        # graph_file_name = f'graph_{index}.pickle'
        #
        # data_aug1 = GINPretrainDataset.read_to_data(self.aug1_folder + graph_file_name)
        # data_aug2 = GINPretrainDataset.read_to_data(self.aug2_folder + graph_file_name)
        # data_aug3 = GINPretrainDataset.read_to_data(self.aug3_folder + graph_file_name)
        #
        # # load and process text
        # text_path = os.path.join(self.text_folder, f'text_{index}.txt')
        #
        # text_list = []
        # count = 0
        # for line in open(text_path, 'r', encoding='utf-8'):
        #     count += 1
        #     text_list.append(line)
        #     if count > 500:
        #         break
        #
        # # print(text_list)
        # if len(text_list) == 1:
        #     three_text_list = [text_list[0], text_list[0][-self.text_max_len:],
        #                        GINPretrainDataset.get_random_substring(text_list[0], self.text_max_len)]
        # elif len(text_list) == 2:
        #     three_text_list = [text_list[0], text_list[1],
        #                        GINPretrainDataset.get_random_substring(text_list[0], self.text_max_len)]
        # else:
        #     three_text_list = random.sample(text_list, 3)
        # text_list.clear()
        #
        # # # load and process text
        # # text_path = os.path.join(self.root, 'text', text_name)
        # # with open(text_path, 'r', encoding='utf-8') as f:
        # #     text_list = f.readlines()
        # # f.close()
        # # # print(text_list)
        # # if len(text_list) < 2:
        # #     two_text_list = [text_list[0], text_list[0][-self.text_max_len:]]
        # # else:
        # #     two_text_list = random.sample(text_list, 2)
        # # text_list.clear()
        #
        # # print(random.sample([1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,18],2))
        # text1, mask1 = self.tokenizer_text(three_text_list[0])
        # text2, mask2 = self.tokenizer_text(three_text_list[1])
        # text3, mask3 = self.tokenizer_text(three_text_list[2])
        #
        # # print(graph_name)
        # # print(text_name)
        #
        # return data_aug1, data_aug2, data_aug3, text1.squeeze(0), mask1.squeeze(0), text2.squeeze(0), \
        #     mask2.squeeze(0), text3.squeeze(0), mask3.squeeze(0)

    # def augment(self, data, graph_aug):
    #     # node_num = data.edge_index.max()
    #     # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
    #     # data.edge_index = torch.cat((data.edge_index, sl), dim=1)
    #
    #     if graph_aug == 'dnodes':
    #         data_aug = drop_nodes(deepcopy(data))
    #     elif graph_aug == 'pedges':
    #         data_aug = permute_edges(deepcopy(data))
    #     elif graph_aug == 'subgraph':
    #         data_aug = subgraph(deepcopy(data))
    #     elif graph_aug == 'mask_nodes':
    #         data_aug = mask_nodes(deepcopy(data))
    #     elif graph_aug == 'random2':  # choose one from two augmentations
    #         n = np.random.randint(2)
    #         if n == 0:
    #             data_aug = drop_nodes(deepcopy(data))
    #         elif n == 1:
    #             data_aug = subgraph(deepcopy(data))
    #         else:
    #             print('sample error')
    #             assert False
    #     elif graph_aug == 'random3':  # choose one from three augmentations
    #         n = np.random.randint(3)
    #         if n == 0:
    #             data_aug = drop_nodes(deepcopy(data))
    #         elif n == 1:
    #             data_aug = permute_edges(deepcopy(data))
    #         elif n == 2:
    #             data_aug = subgraph(deepcopy(data))
    #         else:
    #             print('sample error')
    #             assert False
    #     elif graph_aug == 'random4':  # choose one from four augmentations
    #         n = np.random.randint(4)
    #         if n == 0:
    #             data_aug = drop_nodes(deepcopy(data))
    #         elif n == 1:
    #             data_aug = permute_edges(deepcopy(data))
    #         elif n == 2:
    #             data_aug = subgraph(deepcopy(data))
    #         elif n == 3:
    #             data_aug = mask_nodes(deepcopy(data))
    #         else:
    #             print('sample error')
    #             assert False
    #     else:
    #         data_aug = deepcopy(data)
    #         data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
    #
    #     # if graph_aug == 'dnodes' or graph_aug == 'subgraph' or graph_aug == 'random2' or graph_aug == 'random3' or graph_aug == 'random4':
    #     #     edge_idx = data_aug.edge_index.numpy()
    #     #     _, edge_num = edge_idx.shape
    #     #     idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
    #     #     node_num_aug = len(idx_not_missing)
    #     #     data_aug.x = data_aug.x[idx_not_missing]
    #     #     # data_aug.batch = data.batch[idx_not_missing]
    #     #     idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
    #     #     edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
    #     #     data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
    #
    #     return data_aug

    def tokenizer_text(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=256,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    # mydataset = GraphTextDataset()
    # train_loader = torch_geometric.loader.DataLoader(
    #     mydataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )
    # for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
    #     print(aug1.edge_index.shape)
    #     print(aug1.x.shape)
    #     print(aug1.ptr.size(0))
    #     print(aug2.edge_index.dtype)
    #     print(aug2.x.dtype)
    #     print(text1.shape)
    #     print(mask1.shape)
    #     print(text2.shape)
    #     print(mask2.shape)
    # mydataset = GraphormerPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    # from functools import partial
    # from data_provider.collator import collator_text
    # train_loader = torch.utils.data.DataLoader(
    #         mydataset,
    #         batch_size=8,
    #         num_workers=4,
    #         collate_fn=partial(collator_text,
    #                            max_node=128,
    #                            multi_hop_max_dist=5,
    #                            spatial_pos_max=1024),
    #     )
    # aug1, aug2, text1, mask1, text2, mask2 = mydataset[0]
    mydataset = GINPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    train_loader = torch_geometric.loader.DataLoader(
        mydataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        # persistent_workers = True
    )
    # aug1, aug2, text1, mask1, text2, mask2 = mydataset[0]
    # print(aug1)
    # print(aug1.x.shape)
    # print(aug2)
    # print(aug2.x.dtype)
    # print(text1.shape)
    # print(mask1.shape)
    # print(text2.shape)
    # print(mask2.shape)
    for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
        print(aug1)
        # print(aug1.x.shape)
        # print(aug2)
        # print(aug2.x.dtype)
        # print(text1.shape)
        # print(mask1.shape)

        # print(text2.shape)
        # print(mask2.shape)

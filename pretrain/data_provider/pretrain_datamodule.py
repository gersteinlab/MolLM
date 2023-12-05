# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict

import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data
# from torch_geometric.data.dataloader import Collater
from data_provider.pretrain_dataset import GINPretrainDataset

import sys
import importlib

sys.path.insert(0, '../../../../Transformer-M/Transformer_M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug1: str = 'dnodes',
        graph_aug2: str = 'subgraph',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GINPretrainDataset(root, text_max_len, graph_aug1, graph_aug2)
        # self.pyg_collater = Collater([], [])

    def set_epoch(self, epoch: int):
        self.dataset.set_epoch(epoch % 26)

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    def data_to_graph(self, data):
        new_graph = AttrDict()
        new_graph.update(data.to_dict())
        new_graph = preprocess_item(new_graph)
        return new_graph

    def graph_process(self, graph):
        graph = preprocess_item(graph)
        graph.y = torch.Tensor([[0], [0]]).double()
        return self.convert_to_tensor(graph)

    def number_graph_list(self, graph_list):
        for i, graph in enumerate(graph_list):
            graph.idx = i

    def convert_to_tensor(self, data):
        for key, value in data:
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)

        for key, value in data.items():  # use data.items() to iterate through key-value pairs
            if key in ['attn_bias', 'y', 'pos']:  # float32 keys
                data[key] = value.float()
            elif key in ['idx', 'attn_edge_type', 'spatial_pos', 'in_degree', 'out_degree', 'x', 'edge_input',
                         'node_type_edge']:  # int64 keys
                data[key] = value.long()  # torch.long() corresponds to int64

        return data

    def custom_collate(self, data):
        try:
            if data is None:
                return None

            # One graph in each list per batch item
            augs_1 = []
            augs_2 = []
            augs_3 = []
            augs_4 = []

            # One list per each batch item
            text_list_1 = []
            text_list_2 = []
            text_list_3 = []
            mask_list_1 = []
            mask_list_2 = []
            mask_list_3 = []

            good_tuples = []
            for tuple in data:
                if tuple is None or tuple[0] is None or tuple[1] is None or tuple[2] is None or tuple[3] is None:
                    continue
                good_tuples.append(tuple)

            aug_mask = torch.zeros((len(good_tuples), 4))

            for tuple_idx, tuple in enumerate(good_tuples):
                original_pt = self.graph_process(tuple[0])
                lists = [augs_1, augs_2, augs_3, augs_4]
                for i, aug_list in enumerate(lists):
                    if i < len(tuple[1]):
                        lists[i].append(self.graph_process(tuple[1][i]))
                        aug_mask[tuple_idx][i] = 1
                    else:
                        lists[i].append(original_pt)

                text_list_1.append(tuple[2][0])
                text_list_2.append(tuple[2][1])
                text_list_3.append(tuple[2][2])

                mask_list_1.append(tuple[3][0])
                mask_list_2.append(tuple[3][1])
                mask_list_3.append(tuple[3][2])

            # print('Graph length: ' + str(len(graphs)))
            max_node = 512
            multi_hop_max_dist = 5
            spatial_pos_max = 1024

            self.number_graph_list(augs_1)
            self.number_graph_list(augs_2)
            self.number_graph_list(augs_3)
            self.number_graph_list(augs_4)

            aug1 = collator_3d(augs_1, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                               spatial_pos_max=spatial_pos_max)
            aug2 = collator_3d(augs_2, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                               spatial_pos_max=spatial_pos_max)
            aug3 = collator_3d(augs_3, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                               spatial_pos_max=spatial_pos_max)
            aug4 = collator_3d(augs_4, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
                               spatial_pos_max=spatial_pos_max)

            return aug_mask, aug1, aug2, aug3, aug4, torch.stack(text_list_1), torch.stack(text_list_2), torch.stack(
                text_list_3), \
                torch.stack(mask_list_1), torch.stack(mask_list_2), torch.stack(mask_list_3)
        except Exception as e:
            print(f'Failed collator due to: {e}')
            return None

        # if data is None:
        #     return None
        #
        # graph_lists = []
        # tokens = []
        # masks = []
        # for tuple in data:
        #     if tuple is None or len(tuple) < 3 or tuple[0] is None or tuple[1] is None or tuple[2] is None:
        #         continue
        #     graph_lists.append([preprocess_item(g_list) for g_list in tuple[0]])
        #     tokens.append(tuple[1])
        #     masks.append(tuple[2])
        # # print('Graph length: ' + str(len(graphs)))
        # max_node = 512
        # multi_hop_max_dist = 5
        # spatial_pos_max = 1024
        #
        # if len(graph_lists) == 0 or len(tokens) == 0 or len(masks) == 0:
        #     return None
        #
        # graphs_collated = []
        # for graph_list in graph_lists:
        #     collated_list = []
        #     for graph in graph_list:
        #         self.number_graph_list(graph)
        #         collated_list.append(collator_3d(graph, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist, spatial_pos_max=spatial_pos_max))
        #     graphs_collated.append(collated_list)
        #
        # texts_stacked = [torch.stack(text) for text in tokens]
        # masks_stacked = [torch.stack(mask) for mask in masks]
        #
        # return graphs_collated, texts_stacked, masks_stacked


        # graphs = defaultdict(list)
        # texts = defaultdict(list)
        # masks = defaultdict(list)
        # for aug1, aug2, aug3, text1, mask1, text2, mask2, text3, mask3 in data:
        #     graphs[1].append(self.data_to_graph(aug1))
        #     graphs[2].append(self.data_to_graph(aug2))
        #     graphs[3].append(self.data_to_graph(aug3))
        #     texts[1].append(text1)
        #     texts[2].append(text2)
        #     texts[3].append(text3)
        #     masks[1].append(mask1)
        #     masks[2].append(mask2)
        #     masks[3].append(mask3)
        #
        # # Collate 3d the graphs
        # max_node = 256
        # multi_hop_max_dist = 5
        # spatial_pos_max = 1024
        # self.number_graph_list(graphs[1])
        # self.number_graph_list(graphs[2])
        # self.number_graph_list(graphs[3])
        # aug1 = collator_3d(graphs[1], max_node=max_node, multi_hop_max_dist=multi_hop_max_dist, spatial_pos_max=spatial_pos_max)
        # aug2 = collator_3d(graphs[2], max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
        #                    spatial_pos_max=spatial_pos_max)
        # aug3 = collator_3d(graphs[3], max_node=max_node, multi_hop_max_dist=multi_hop_max_dist,
        #                    spatial_pos_max=spatial_pos_max)
        #
        # return aug1, aug2, aug3, torch.stack(texts[1]), torch.stack(masks[1]), \
        #     torch.stack(texts[2]), torch.stack(masks[2]), torch.stack(texts[3]), torch.stack(masks[3])

    def train_dataloader(self):
        # loader = torch_geometric.data.DataLoader(
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.custom_collate,
            persistent_workers=True
            # timeout=99999
            # persistent_workers = True  # Set later..
        )
        print('len(train_dataloader)', len(loader))
        return loader
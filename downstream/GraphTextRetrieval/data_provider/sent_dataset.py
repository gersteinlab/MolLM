import gc
import pickle
import random

import torch
from torch_geometric.data import Dataset
import numpy as np
import os
from transformers import BertTokenizer


def drop_nodes(graph, drop_percentage=0.1):
    num_nodes = len(graph['x'])
    num_nodes_to_drop = int(drop_percentage * num_nodes)
    nodes_to_drop = random.sample(range(num_nodes), num_nodes_to_drop)

    # Remove dropped nodes
    x = graph['x']
    pos = graph['pos']
    graph['x'] = torch.tensor([x[i] for i in range(num_nodes) if i not in nodes_to_drop])
    graph['pos'] = torch.tensor([pos[i] for i in range(num_nodes) if i not in nodes_to_drop])

    # Remove dropped node edges and update edge indices
    edge_index = graph['edge_index']
    edge_attr = graph['edge_attr']
    edge_mapping = {old: new for new, old in enumerate([i for i in range(num_nodes) if i not in nodes_to_drop])}

    new_edge_index = []
    new_edge_attr = []
    for idx, (src, dest) in enumerate(edge_index.t()):
        if src.item() not in nodes_to_drop and dest.item() not in nodes_to_drop:
            new_edge_index.append([edge_mapping[src], edge_mapping[dest]])
            new_edge_attr.append(edge_attr[idx])

    graph['edge_index'] = torch.tensor(new_edge_index).t()
    graph['edge_attr'] = torch.tensor(new_edge_attr)

    return graph


class GINSentDataset(Dataset):
    def __init__(self, root, args):
        super(GINSentDataset, self).__init__(root)
        self.root = root
        self.text_max_len = args.text_max_len
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        #self.cor.append(cnt)
        for text_name in self.text_name_list:
            text_path = os.path.join(self.root, 'text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        np.save('output/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

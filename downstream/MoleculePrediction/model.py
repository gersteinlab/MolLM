import importlib
import sys
from types import SimpleNamespace

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

sys.path.insert(0, '../graph-transformer')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__(aggr = "add")
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes = x.size(0))
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin", use_3d = False):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        print('=== 2D/3D SETTING ===')
        print('Transformer-MoMu: ' + ('USING 3D..' if use_3d else 'USING 2D ONLY..'))
        print('=== ------------- ===')
        args = SimpleNamespace(_name='transformer_m_base', act_dropout=0.3, activation_fn='gelu', add_3d=use_3d,
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

        self.transformer = TransformerM(args=args)  # type: pytorch_lightning.LightningModule

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        # if self.JK == "concat":
        #     self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        # else:
        #     self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

        # .61
        self.graph_pred_head = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),  # Adding 20% dropout
            torch.nn.Linear(300, self.num_tasks),
        )
        self.xavier_uniform_zeros_(self.graph_pred_head)

        # self.graph_pred_head = torch.nn.Linear(768, self.num_tasks)

    def xavier_uniform_zeros_(self, layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)

        # if 'pretrain' in model_file:
        #     if model_file == 'pretrain_MoMu-S':
        #         ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt")
        #     elif model_file == 'pretrain_MoMu-K':
        #         ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt")
        #     ckpt = ckpt['state_dict']
        #     pretrained_dict = {k[14:]: v for k, v in ckpt.items()}
        #     missing_keys, unexpected_keys = self.gnn.load_state_dict(pretrained_dict, strict=False)
        #     # print(missing_keys)
        #     # print(unexpected_keys)
        # else:
        #     self.load_state_dict(torch.load("./GIN_checkpoints/"+model_file))
        ckpt = torch.load(f'all_checkpoints/{model_file}')['state_dict']
        ckpt = {key.replace("graph_encoder.", ""): value for key, value in ckpt.items() if key.startswith("graph_encoder.")}
        # print('Existing keys: ' + str(ckpt.keys()))
        missing_keys, unexpected_keys = self.transformer.load_state_dict(ckpt, strict=False)
        print('Missing keys: ' + str(missing_keys))
        print('Unexpected keys: ' + str(unexpected_keys))

    def forward_through_graph_encoder(self, collated):
        inner_states, atom_output = self.transformer.molecule_encoder(
            collated,
            segment_labels=None,
            perturb=None,
            last_state_only=True
        )

        del atom_output
        torch.cuda.empty_cache()

        last_state = inner_states[0]
        molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
        return molecule_embedding

    def forward(self, batch):
        molecule_embedding = self.forward_through_graph_encoder(batch)
        return self.graph_pred_head(molecule_embedding)

    def forward_both(self, batch):
        molecule_embedding = self.forward_through_graph_encoder(batch)
        return molecule_embedding, self.graph_pred_head(molecule_embedding)


if __name__ == "__main__":
    
    pass


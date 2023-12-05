import gc
import traceback
from types import SimpleNamespace
from typing import Union, IO, Optional, Dict, Callable

import pytorch_lightning
import torch
import torch.nn as nn
import torch_scatter

# from model.gin_model import GNN
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import sys
import importlib

sys.path.insert(0, '/gpfs/slayman/pi/gerstein/xt86/ismb2023/Transformer-MoMu/Transformer-M/')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")

TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d

use_3d = False


def xavier_uniform_zeros_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GINSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            gin_hidden_dim,
            gin_num_layers,
            drop_ratio,
            graph_pooling,
            graph_self,
            bert_hidden_dim,
            bert_pretrain,
            projection_dim,
            lr,
            weight_decay,
            initialize_transformerm
    ):
        global use_3d
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_num_layers = gin_num_layers
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self.graph_self = graph_self

        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain

        self.projection_dim = projection_dim

        self.lr = lr
        self.weight_decay = weight_decay

        self.initialize_transformerm = initialize_transformerm
        # self.m_dropout = m_dropout

        # self.graph_encoder = GNN(
        #     num_layer=self.gin_num_layers,
        #     emb_dim=self.gin_hidden_dim,
        #     gnn_type='gin',
        #     # virtual_node=True,
        #     # residual=False,
        #     drop_ratio=self.drop_ratio,
        #     JK='last',
        #     # graph_pooling=self.graph_pooling,
        # )

        # Use 3D set in main.py
        print('=== 2D/3D SETTING ===')
        # print('Transformer-MoMu: ', ('USING 3D..' if use_3d else 'USING 2D ONLY..'))
        # print(f'DROPOUT = {self.dropout}')
        # dropouts = [float(s) for s in self.m_dropout.split(',')]
        print('=== ------------- ===')
        args = SimpleNamespace(_name='transformer_m_base', act_dropout=0.1, activation_fn='gelu', add_3d=use_3d,
                               add_prev_output_tokens=False, all_gather_list_size=16384, amp=False, amp_batch_retries=2,
                               amp_init_scale=128, amp_scale_window=None, apply_init=True, arch='transformer_m_base',
                               attention_dropout=0.1, azureml_logging=False, batch_size=256, batch_size_valid=256,
                               best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False,
                               bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0,
                               combine_valid_subsets=None, cpu=False, cpu_offload=False, criterion='graph_prediction',
                               curriculum=0, data_buffer_size=20, data_path='NOT-IN-USE', dataset_impl=None,
                               dataset_name='NOT-IN-USE', ddp_backend='legacy_ddp', ddp_comm_hook='none',
                               device_id=0, disable_validation=False, distributed_backend='nccl',
                               distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=2,
                               distributed_port=-1, distributed_rank=0, distributed_world_size=2, dropout=0,
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
                               max_valid_steps=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False,
                               memory_efficient_fp16=False, metric='mae', min_loss_scale=0.0001,
                               mode_prob='0.2,0.2,0.6', model_parallel_size=1, multi_hop_max_dist=5, no_2d=False,
                               no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False,
                               no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False,
                               no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False,
                               noise_scale=0.2, not_fsdp_flatten_parameters=False, nprocs_per_node=2,
                               num_3d_bias_kernel=128, num_atoms=4608, num_classes=1, num_edge_dis=128, num_edges=1536,
                               num_in_degree=512, num_out_degree=512, num_segment=2, num_shards=1, num_spatial=512,
                               num_workers=16, on_cpu_convert_precision=False, optimizer=None, optimizer_overrides='{}',
                               pad=1, patience=-1, pipeline_balance=None, pipeline_checkpoint='never',
                               pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None,
                               pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None,
                               pipeline_model_parallel=False, plasma_path='/tmp/plasma', pooler_activation_fn='tanh',
                               profile=False, quantization_config_path=None, required_batch_size_multiple=8,
                               required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False,
                               reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False,
                               restore_file='checkpoint_last.pt', sandwich_ln=False, save_dir='NOT-IN-USE',
                               save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sent_loss=False,
                               sentence_avg=False, sentence_class_num=2, separator_token=None, shard_id=0,
                               share_encoder_input_output_embed=False, shorten_data_split_list='',
                               shorten_method='none', simul_type=None, skip_invalid_size_inputs_valid_test=False,
                               slowmo_base_algorithm='localsgd', slowmo_momentum=None, split='valid', stop_min_lr=-1.0,
                               stop_time_hours=0, store_ema=False, suppress_crashes=False, task='NOT-IN-USE',
                               tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, tpu=False,
                               train_subset='train', unk=3, update_epoch_batch_itr=False, update_freq=[1],
                               update_ordered_indices_seed=False, use_bmuf=False, use_plasma_view=False,
                               use_sharded_state=False,
                               user_dir='NOT-IN-USE',
                               valid_subset='valid', validate_after_updates=0, validate_interval=1,
                               validate_interval_updates=0, wandb_project=None, warmup_updates=0,
                               write_checkpoints_asynchronously=False, zero_sharding='none')

        self.graph_encoder = TransformerM(args=args)  # type: pytorch_lightning.LightningModule
        if self.initialize_transformerm:
            print('Initializing graph transformer with L12.pt')
            model_state = torch.load('logs/L12/L12.pt')["model"]
            # Remove "encoder." from start of key
            model_state = {weight_name.removeprefix("encoder."): v for weight_name, v in model_state.items()}
            missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(
                model_state, strict=False
            )
            del model_state

            print('Missing Keys: ', str(missing_keys))
            print('Unexpected Keys: ', str(unexpected_keys))

        if self.bert_pretrain:
            # NOT USING SCIBERT
            self.text_encoder = TextEncoder(pretrained=False)
        else:
            # USING SCIBERT
            self.text_encoder = TextEncoder(pretrained=True)

        if self.bert_pretrain:
            # NOT USING SCIBERT, use KV-PLM
            print("bert load kvplm")
            ckpt = torch.load('kvplm_pretrained/ckpt_KV_1.pt')
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
            else:
                pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            # print(pretrained_dict.keys())
            # print(self.text_encoder.state_dict().keys())
            self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # missing_keys, unexpected_keys = self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)
            print("bert load kvplm DONE")
        # self.feature_extractor.freeze()

        self.graph_proj_head = nn.Sequential(
          nn.Linear(768, 768),
          nn.ReLU(inplace=True),
          nn.Linear(768, 256)
        )
        self.graph_proj_head.apply(xavier_uniform_zeros_)
        # self.graph_proj_head = nn.Linear(768, 256)
        # torch.nn.init.xavier_uniform(self.graph_proj_head.weight)
        # torch.nn.init.zeros_(self.graph_proj_head.bias)
        self.text_proj_head = nn.Sequential(
          nn.Linear(768, 768),
          nn.ReLU(inplace=True),
          nn.Linear(768, 256)
        )
        self.text_proj_head.apply(xavier_uniform_zeros_)
        # self.text_proj_head = nn.Linear(768, 256)
        # torch.nn.init.xavier_uniform(self.text_proj_head.weight)
        # torch.nn.init.zeros_(self.text_proj_head.bias)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward_original(self, features_graph, features_text, aug_mask):
        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        # Only keep logits for which the corresponding aug_mask value is 1
        # if aug_mask is not None:
        #     mask = aug_mask.bool()
        #     logits_per_graph = logits_per_graph[mask]
        #     logits_per_text = logits_per_text[mask]

        labels = torch.arange(logits_per_graph.size(0), dtype=torch.long, device=self.device)  # size is B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def forward(self, collated, text, mask):
        inner_states, atom_output = self.graph_encoder.molecule_encoder(
            collated,
            segment_labels=None,
            perturb=None,
            last_state_only=True
        )

        del atom_output
        torch.cuda.empty_cache()

        last_state = inner_states[0]
        molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
        return self.graph_proj_head(molecule_embedding), self.text_proj_head(self.text_encoder(text, mask))

    def average_losses(self, losses):
        """
        Averages a list of scalar tensors, ignoring NaN values.

        Args:
        losses (list of torch.Tensor): List of scalar tensors representing losses.

        Returns:
        mean_loss (torch.Tensor): Mean of the losses, ignoring NaNs.
        """
        # Convert list to tensor
        losses_tensor = torch.stack(losses)

        # Create a mask of non-NaN values
        mask = ~torch.isnan(losses_tensor)

        # Select non-NaN values and calculate mean
        mean_loss = torch.masked_select(losses_tensor, mask).mean()

        return mean_loss

    def training_step(self, batch, batch_idx):
        try:
            aug_mask, aug1, aug2, aug3, aug4, text1, text2, text3, mask1, mask2, mask3 = batch

            graph1_rep, text1_rep = self.forward(aug1, text1, mask1)
            graph2_rep, text2_rep = self.forward(aug2, text2, mask2)
            graph3_rep, text3_rep = self.forward(aug3, text3, mask3)
            graph4_rep, text3_rep = self.forward(aug4, text3, mask3)

            # 4 * 3 = 12 normal losses
            _, _, loss11 = self.forward_original(graph1_rep, text1_rep, aug_mask[:, 0])
            _, _, loss12 = self.forward_original(graph1_rep, text2_rep, aug_mask[:, 0])
            _, _, loss13 = self.forward_original(graph1_rep, text3_rep, aug_mask[:, 0])

            _, _, loss21 = self.forward_original(graph2_rep, text1_rep, aug_mask[:, 1])
            _, _, loss22 = self.forward_original(graph2_rep, text2_rep, aug_mask[:, 1])
            _, _, loss23 = self.forward_original(graph2_rep, text3_rep, aug_mask[:, 1])

            _, _, loss31 = self.forward_original(graph3_rep, text1_rep, aug_mask[:, 2])
            _, _, loss32 = self.forward_original(graph3_rep, text2_rep, aug_mask[:, 2])
            _, _, loss33 = self.forward_original(graph3_rep, text3_rep, aug_mask[:, 2])

            _, _, loss41 = self.forward_original(graph4_rep, text1_rep, aug_mask[:, 3])
            _, _, loss42 = self.forward_original(graph4_rep, text2_rep, aug_mask[:, 3])
            _, _, loss43 = self.forward_original(graph4_rep, text3_rep, aug_mask[:, 3])

            if self.graph_self:
                # 4 choose 2 = 6 self losses
                _, _, loss_graph_self_1 = self.forward_original(graph1_rep, graph2_rep, None)
                _, _, loss_graph_self_2 = self.forward_original(graph2_rep, graph3_rep, None)
                _, _, loss_graph_self_3 = self.forward_original(graph1_rep, graph3_rep, None)
                _, _, loss_graph_self_4 = self.forward_original(graph1_rep, graph4_rep, None)
                _, _, loss_graph_self_5 = self.forward_original(graph2_rep, graph4_rep, None)
                _, _, loss_graph_self_6 = self.forward_original(graph3_rep, graph4_rep, None)

                loss_1 = self.average_losses([loss11, loss12, loss13, loss21 , loss22 , loss23 , loss31 , loss32 , loss33 , loss41 , loss42 , loss43])
                loss_2 = self.average_losses([loss_graph_self_1, loss_graph_self_2, loss_graph_self_3, loss_graph_self_4, loss_graph_self_5, loss_graph_self_6])

                loss = loss_1 + loss_2
            else:
                loss = self.average_losses([loss11, loss12, loss13, loss21, loss22, loss23, loss31, loss32, loss33])

            self.log("train_loss", loss)
            return loss
        except Exception as e:
            print(traceback.format_exc())
            print(f'Failed training step, returning loss of 0 to avoid gradients changing: {e}')
            if 'CUDA out' in f'{e}':
                gc.collect()
                torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=768)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--graph_pooling', type=str, default='sum')
        parser.add_argument('--graph_self', action='store_true', help='use graph self-supervise or not', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', action='store_false', default=True)
        parser.add_argument('--projection_dim', type=int, default=256)
        # optimization
        parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')

        parser.add_argument('--initialize_transformerm', action='store_true', default=False)
        parser.add_argument('--use_3d', action='store_true', default=False)
        return parent_parser


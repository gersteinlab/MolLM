import importlib
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from model.gin_model import GNN


sys.path.insert(0, '../../graph-transformer')
# sys.path.insert(0, '/data/code/Transformer-M')
transformerm_models = importlib.import_module("Transformer_M.models")
TransformerM = transformerm_models.TransformerM
data = importlib.import_module("Transformer_M.data")
preprocess_item = data.wrapper.preprocess_item
collator_3d = data.collator_3d


class GinDecoder(nn.Module):
    def __init__(self, has_graph=True, MoMuK=True, model_size='base', use_3d=True):
        super(GinDecoder, self).__init__()
        self.has_graph = has_graph
        self.main_model = T5ForConditionalGeneration.from_pretrained("molt5-"+model_size+"-smiles2caption/")
        print(f'hidden_size: {self.main_model.config.hidden_size},\
                d_model: {self.main_model.config.d_model},\
                num_decoder_layers: {self.main_model.config.num_decoder_layers},\
                num_heads: {self.main_model.config.num_heads},\
                d_kv: {self.main_model.config.d_kv}')

        for p in self.main_model.named_parameters():
            p[1].requires_grad = False

        if has_graph:
            # self.graph_encoder = GNN(
            #     num_layer=5,
            #     emb_dim=300,
            #     gnn_type='gin',
            #     drop_ratio=0.0,
            #     JK='last',
            # )
            print('=== 2D/3D SETTING ===')
            print('Transformer-MoMu: ' + ('USING 3D..' if use_3d else 'USING 2D ONLY..'))
            # print('DROPOUT BUMPED TO 0.1')
            print('=== ------------- ===')
            args = SimpleNamespace(_name='transformer_m_base', act_dropout=0.1, activation_fn='gelu', add_3d=use_3d,
                                   add_prev_output_tokens=False, all_gather_list_size=16384, amp=False,
                                   amp_batch_retries=2,
                                   amp_init_scale=128, amp_scale_window=None, apply_init=True,
                                   arch='transformer_m_base',
                                   attention_dropout=0.1, azureml_logging=False, batch_size=256, batch_size_valid=256,
                                   best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False,
                                   bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0,
                                   combine_valid_subsets=None, cpu=False, cpu_offload=False,
                                   criterion='graph_prediction',
                                   curriculum=0, data_buffer_size=20, data_path='NOT-IN-USE', dataset_impl=None,
                                   dataset_name='NOT-IN-USE', ddp_backend='legacy_ddp', ddp_comm_hook='none',
                                   device_id=0, disable_validation=False, distributed_backend='nccl',
                                   distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=2,
                                   distributed_port=-1, distributed_rank=0, distributed_world_size=2, dropout=0.15,
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

            self.molecule_encoder = TransformerM(args=args)  # type: pytorch_lightning.LightningModule
            
            # if MoMuK:
            #     ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt")
            # else:
            #     ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt")
            ckpt = torch.load('./M3_checkpoints/model-epoch=394.ckpt')

            ckpt = ckpt['state_dict']

            new_ckpt = {}
            for k, v in ckpt.items():
                if 'text_encoder' in k:
                    continue
                new_key = k.removeprefix('graph_encoder.')
                new_ckpt[new_key] = v

            # pretrained_dict = {k[14:]: v for k, v in ckpt.items()}
            missing_keys, unexpected_keys = self.molecule_encoder.load_state_dict(new_ckpt, strict=False)
            print('Missing keys: ' + str(missing_keys))
            print('Unexpected keys: ' + str(unexpected_keys))

            # Andrew comment: not training the graph encoder??
            for p in self.molecule_encoder.named_parameters():
                p[1].requires_grad = False
            
            self.graph_projector = nn.Sequential(
                nn.Linear(768, self.main_model.config.hidden_size), # Andrew: 300 -> 768
                nn.ReLU(inplace=True),
                nn.Linear(self.main_model.config.hidden_size, self.main_model.config.hidden_size)
            )
            # self.graph_projector_dropout = torch.nn.Dropout(0.2)
            # self.graph_projector = nn.Linear(300, self.main_model.config.hidden_size)

    def forward_through_graph_encoder(self, collated):
        inner_states, atom_output = self.molecule_encoder.molecule_encoder(
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

    def forward(self, arg1, arg2, arg3, arg4, arg5=None):
        if arg5 is not None:
            return self.forward_norm(arg1, arg2, arg3, arg4, arg5)
        else:
            return self.translate(arg1, arg2, arg3, arg4)

    def forward_norm(self, batch, input_ids, encoder_attention_mask, decoder_attention_mask, label):
        try:
            device = encoder_attention_mask.device
            B, _ = encoder_attention_mask.shape

            smiles_embeds = self.main_model.encoder(input_ids=input_ids,
                                                    attention_mask=encoder_attention_mask).last_hidden_state

            if self.has_graph:
                graph_rep = self.forward_through_graph_encoder(batch)
                graph_rep = self.graph_projector(graph_rep)
                smiles_embeds = torch.cat([graph_rep.unsqueeze(1), smiles_embeds], dim=1)
                encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask], dim=1)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=smiles_embeds,
                hidden_states=None,
                attentions=None,
            )
            loss = self.main_model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=label
            ).loss

            return loss
        except Exception as e:
            print(f'Failed GinT5 forward: {e}')
            return torch.tensor(0.0, requires_grad=True)


    def translate(self, batch, input_ids, encoder_attention_mask, tokenizer):
        B = 0
        try:
            device = encoder_attention_mask.device
            B, _ = encoder_attention_mask.shape

            smiles_embeds = self.main_model.encoder(input_ids=input_ids,
                                                    attention_mask=encoder_attention_mask).last_hidden_state

            if self.has_graph:
                graph_rep = self.forward_through_graph_encoder(batch)
                graph_rep = self.graph_projector(graph_rep)
                smiles_embeds = torch.cat([graph_rep.unsqueeze(1), smiles_embeds], dim=1)
                encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask], dim=1)

            # input_prompt = ["The molecule is"] * B
            # decoder_input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
            # decoder_input_ids = decoder_input_ids.to(device)
            num_beams = 5

            encoder_outputs = BaseModelOutput(
                last_hidden_state=smiles_embeds,
                hidden_states=None,
                attentions=None,
            )
            outputs = self.main_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,  # important
                num_beams=num_beams,
                max_length=512,
                # eos_token_id=self.main_model.config.eos_token_id,
                # decoder_start_token_id=self.main_model.config.decoder_start_token_id,
                # decoder_input_ids = decoder_input_ids,
            )

            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return res
            # return [str_to_tensor(s) for s in res]
            # return outputs
        except Exception as e:
            print(f'Failed GinT5 translate: {e}')
            return [''] * B

    def gin_encode(self, batch):
        node_reps = self.molecule_encoder(batch)
        return node_reps


def _num_to_str(nums):
    s = ''
    for batch in nums:
        for char in batch:
            s += chr(char)
    return s


def _str_to_num(string, batch_size=64):
    "Encodes `string` to a decodeable number and breaks it up by `batch_size`"
    batch, inner_batch = [], []
    for i, char in enumerate(string):
        char = ord(char)
        inner_batch.append(char)
        if (len(inner_batch) == batch_size) or (i == len(string) - 1):
            batch.append(inner_batch)
            inner_batch = []
    return batch


def str_to_tensor(string) -> torch.tensor:
    """
    Encodes `string` to a tensor of shape [1,N,batch_size] where
    `batch_size` is the number of characters and `n` is
    (len(string)//batch_size) + 1
    """
    return torch.tensor(_str_to_num(string), dtype=torch.long)


def tensor_to_str(x:torch.Tensor) -> str:
    """
    Decodes `x` to a string. `x` must have been encoded from
    `str_to_tensor`
    """
    return _num_to_str(x.tolist())


if __name__ == '__main__':
    model = T5ForConditionalGeneration.from_pretrained("molt5_base/")
    for p in model.named_parameters():
        if 'lm_head' in p[0] or 'shared' in p[0]:
	        print(p[1])
    
    print(model.shared)
    print(model.lm_head)
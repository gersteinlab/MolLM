#!/bin/bash
# finetune MoMu and save as 'finetune_save/finetune_para.pt '
export CUDA_VISIBLE_DEVICES=1
python main.py --batch_size 8 --init_checkpoint "all_checkpoints/model-epoch=141.ckpt" --output finetune_save/finetune_m3_sent_b16 --data_type 1 --if_test 0 --if_zeroshot 0

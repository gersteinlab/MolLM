#!/bin/bash
# finetune MoMu and save as 'finetune_save/finetune_para.pt '
export CUDA_VISIBLE_DEVICES=0
python main.py --batch_size 8 --init_checkpoint "all_checkpoints/model-epoch=141.ckpt" --output finetune_save/finetune_m3_para_b16 --data_type 0 --if_test 0 --if_zeroshot 0
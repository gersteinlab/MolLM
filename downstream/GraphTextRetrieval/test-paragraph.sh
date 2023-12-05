#!/bin/bash
# finetune MoMu and save as 'finetune_save/finetune_para.pt '
export CUDA_VISIBLE_DEVICES=2
python main.py --batch_size 64 --init_checkpoint "all_checkpoints/model-epoch=141.ckpt" --output finetune_save/finetune_m3_para_b16-dev0.96-22.pt --data_type 0 --if_test 2 --if_zeroshot 0

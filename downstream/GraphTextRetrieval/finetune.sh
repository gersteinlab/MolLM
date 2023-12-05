#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gin.py --batch_size=64 --accelerator='gpu' --gpus='0,1,2,3' --graph_self --max_epochs=300 --num_workers=8
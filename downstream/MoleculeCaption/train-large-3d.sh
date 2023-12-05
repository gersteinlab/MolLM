#!/bin/sh
CUDA_VISIBLE_DEVICES=5
WANDB_API_KEY=e6ad08a8e80e50e3195d3e1d27ce9ebfd8ef0997 python main_transformer_smiles2caption.py --mode train --model_size large --MoMuK --use_3d
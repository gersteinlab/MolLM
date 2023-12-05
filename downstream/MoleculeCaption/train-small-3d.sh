#!/bin/sh
CUDA_VISIBLE_DEVICES=1
python main_transformer_smiles2caption.py --mode train --model_size small --MoMuK --use_3d
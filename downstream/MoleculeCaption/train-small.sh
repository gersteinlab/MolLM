#!/bin/sh
CUDA_VISIBLE_DEVICES=0
python main_transformer_smiles2caption.py --mode train --model_size small --MoMuK
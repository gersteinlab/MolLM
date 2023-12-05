#!/bin/sh
CUDA_VISIBLE_DEVICES=2
python main_transformer_smiles2caption.py --mode train --model_size base --MoMuK
#!/bin/sh
CUDA_VISIBLE_DEVICES=4
python main_transformer_smiles2caption.py --mode train --model_size large --MoMuK
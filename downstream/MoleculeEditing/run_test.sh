#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python edit_molecule.py \
  --input_file test.txt \
  --attempts 2 \
  --input_text "This molecule is insoluble in water." \
  --property solubility \
  --goal decrease \
  --lr 0.02 \
#  --temperature 0.075
#  --lr_scheduler_steps 250 \
#  --total_steps 500
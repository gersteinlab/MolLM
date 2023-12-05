#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python edit_molecule.py \
  --input_text "This molecule is insoluble in water." \
  --property solubility \
  --goal decrease

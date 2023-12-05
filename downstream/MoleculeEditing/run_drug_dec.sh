#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python edit_molecule.py \
  --input_text "This molecule is not like a drug" \
  --property likeness \
  --goal decrease

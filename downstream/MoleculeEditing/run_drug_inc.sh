#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python edit_molecule.py \
  --input_text "This molecule is like a drug" \
  --property likeness \
  --goal increase

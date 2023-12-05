#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python edit_molecule.py \
  --input_text "This molecule has low permeability" \
  --property permeability \
  --goal decrease

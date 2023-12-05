#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python edit_molecule.py \
  --input_text "This molecule is soluble in water." \
  --property solubility \
  --goal increase

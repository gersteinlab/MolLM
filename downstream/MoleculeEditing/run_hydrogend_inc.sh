#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python edit_molecule.py \
  --input_text "This molecule has more hydrogen bond donors." \
  --property hydrogen_donor \
  --goal increase

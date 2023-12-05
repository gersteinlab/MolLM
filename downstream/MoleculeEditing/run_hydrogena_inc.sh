#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python edit_molecule.py \
  --input_text "This molecule has more hydrogen bond acceptors." \
  --property hydrogen_acceptor \
  --goal increase

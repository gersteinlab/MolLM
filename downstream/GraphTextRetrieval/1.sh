#!/bin/sh
# zeroshot testing on phy_data with paragraph-level:
python main.py --init_checkpoint "all_checkpoints/model-epoch=117.ckpt" --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
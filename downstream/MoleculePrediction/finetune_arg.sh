#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset=$1
cuda_device=-1

case $dataset in
    tox21) cuda_device=0;;
    hiv) cuda_device=1;;
    pcba) cuda_device=2;;
    muv) cuda_device=3;;
    bace) cuda_device=4;;
    toxcast) cuda_device=5;;
    sider) cuda_device=6;;
    clintox) cuda_device=7;;
    *) echo "Invalid dataset. Please choose from: tox21, hiv, pcba, muv, bace, toxcast, sider, clintox"; exit 1;;
esac

cuda_device=0

export CUDA_VISIBLE_DEVICES=$cuda_device

python finetune.py --input_model_file model-epoch=394.ckpt --runseed 0 --dataset $dataset

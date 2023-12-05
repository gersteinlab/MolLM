#!/bin/bash
# Log everything
current_datetime=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="train_$current_datetime.log"
exec > >(tee -a "$log_file") 2>&1

# Create checkpoints folder
checkpoints_folder="./all_checkpoints/pretrain_transformerm_all/"
mkdir -p $checkpoints_folder

# Run
while true; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gin.py --batch_size=108 --accelerator='gpu' --graph_self --max_epochs=2000 --num_workers=6 --initialize_transformerm --use_3d
    killall python
    sleep 10
done

## Loop for all epochs
#while true; do
#    # Count completed epochs
#    folder_count=$(find "$checkpoints_folder" -mindepth 1 -maxdepth 1 -type d | wc -l)
#
#    next_epoch=$((folder_count + 1))
#
#    # Stop after epoch 300
#    if [ $next_epoch -gt 300 ]; then
#        break
#    fi
#
#    # Train
#    echo "=== Training Epoch ${next_epoch}... ==="
#    echo $folder_count > current_epoch
#    WANDB_API_KEY=e6ad08a8e80e50e3195d3e1d27ce9ebfd8ef0997 CUDA_VISIBLE_DEVICES=0,1 python train_gin.py --batch_size=128 --accelerator='gpu' --graph_self --max_epochs=1 --num_workers=4 --epoch $next_epoch --initialize_transformerm
#done

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 gdb --args python train_gin.py --batch_size=32 --accelerator='gpu' --graph_self --max_epochs=1 --num_workers=0 --epoch 1 --initialize_transformerm

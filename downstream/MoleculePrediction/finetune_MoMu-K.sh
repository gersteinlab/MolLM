#### GIN fine-tuning

for dataset in bbbp tox21 toxcast sider clintox muv hiv bace
do
python finetune.py --input_model_file pretrain_MoMu-K --runseed 0 --dataset $dataset
done
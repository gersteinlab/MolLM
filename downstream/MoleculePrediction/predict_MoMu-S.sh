#### GIN predict

for dataset in bbbp tox21 toxcast sider clintox muv hiv bace
do
python predict.py --input_model_file MoMu-S.pth --dataset $dataset
done


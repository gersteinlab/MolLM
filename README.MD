# MolLM: A Unified Language Model to Integrate Biomedical Text with 2D and 3D Molecular Representations


<p align="center">
   📖 <a href="https://www.biorxiv.org/content/10.1101/2023.11.25.568656v3" target="_blank">Paper</a>  
</p>


This is code for the [MolLM model](https://www.biorxiv.org/content/10.1101/2023.11.25.568656v2). The code is organized as follows:
- `dataset-creation`: Code to generate our dataset of 160k graph text pairs
- `downstream`: Code for the MolLM and downstream tasks
- `environments`: Conda environments 
  - Use `base` environment for general model usage and pretraining
  - Use other specific environments for each downstream tasks
- `pretrain`: Code for pretraining MolLM

Large dataset and model checkpoints files are in zip files downloadable from this [Google Drive folder](https://drive.google.com/drive/folders/17XhqdsDOxiT8PEDLHdsLPKf62PXPmbms?usp=sharing). The paths within this folder correspond to locations where the zip folders should be decompressed. 

## Pre-training

Utilize the training script `pretrain/train.sh` to pretrain the model. Inside the script, the GPUs used, batch size, epochs, 3D setting, and checkpoints save location may be viewed and modified as desired.

## Downstream Tasks

See the above on environments for the appropriate Conda environment to use for each downstream task.

For graph-text cross-modality retrieval, within the `downstream/GraphTextRetrieval` folder, the `finetune-sentence.sh` and `finetune-paragraph.sh` scripts are for finetuning the moel for the task at the sentence and paragraph respectively. Similarly, the `test-sent.sh` and `test-paragraph.sh` scripts perform the evaluation. These scripts can be modified to change the pretrained epoch and GPUs used.

For molecule captioning, within the `downstream/MoleculeCaption` folder, the scripts starting with `train-` are for training the small & base versions in both 2D & 3D settings. The `test-base.sh` provides an example of utilizing the evaluation script. These scripts can be modified to change the pretrained checkpoints (for both MolT5 and MolLM) and GPUs used.

For molecule editing, within the `downstream/MoleculeEditing` folder, the scripts starting with `run_` are for generating the molecules for various prompts. The `eval.py` takes in the generated molecule file, desired metric, desired change in the metric, and output path to output an evaluation of the generation. These scripts can be modified to change the pretrained checkpoint, prompts, and GPUs used.

Finally, for molecule prediction, within the `downstream/MoleculePrediction` folder, the `finetune_arg.sh` script takes in a MoleculeNet dataset as an argument to finetune the model and also reports the evaluation scores. These scripts can be modified to change the pretrained checkpoint and GPUs used.

## Cite us
```
@article{tang2023mollm,
  title={MolLM: A Unified Language Model to Integrate Biomedical Text with 2D and 3D Molecular Representations},
  author={Tang, Xiangru and Tran, Andrew and Tan, Jeffrey and Gerstein, Mark},
  journal={bioRxiv},
  pages={2023--11},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

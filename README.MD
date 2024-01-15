# MolLM: A Unified Language Model to Integrate Biomedical Text with 2D and 3D Molecular Representations


<p align="center">
   📖 <a href="https://www.biorxiv.org/content/10.1101/2023.11.25.568656v2" target="_blank">Paper</a>  
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

## Downstream Tasks

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

# MolLM
This is code for the [MolLM model](https://www.biorxiv.org/content/10.1101/2023.11.25.568656v1). The code is organized as follows:
- `dataset-creation`: Code to generate our dataset of 160k graph text pairs
- `downstream`: Code for the MolLM and downstream tasks
- `environments`: Conda environments 
  - Use `base` environment for general model usage and pretraining
  - Use other specific environments for each downstream tasks
- `pretrain`: Code for pretraining MolLM

Large dataset and model checkpoints files are in zip files downloadable from this [Google Drive folder](https://drive.google.com/drive/folders/17XhqdsDOxiT8PEDLHdsLPKf62PXPmbms?usp=sharing). The paths within this folder correspond to locations where the zip folders should be decompressed. 


## Cite us
```
@article{tang2023mollm,
  title={MolLM: Integrating 3D and 2D Molecular Representations with Biomedical Text via a Unified Pre-trained Language Model},
  author={Tang, Xiangru and Tran, Andrew and Tan, Jeffrey and Gerstein, Mark},
  journal={bioRxiv},
  pages={2023--11},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

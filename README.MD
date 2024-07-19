# MolLM: A Unified Language Model to Integrate Biomedical Text with 2D and 3D Molecular Representations


<p align="center">
   📖 <a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i357/7700902" target="_blank">Paper</a>  
</p>


This is code for the [MolLM model](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i357/7700902). The code is organized as follows:
- `dataset-creation`: Code to generate our dataset of 160k graph text pairs
- `downstream`: Code for the MolLM and downstream tasks
- `environments`: Conda environments 
  - Use `base` environment for general model usage and pretraining
  - Use other specific environments for each downstream tasks
- `pretrain`: Code for pretraining MolLM

Model checkpoints files are in zip files downloadable from this [Hugging Face model](https://huggingface.co/andrewt28/MolLM/tree/main). The paths within this folder correspond to locations where the zip folders should be decompressed. The dataset is available at this [Hugging Face dataset](https://huggingface.co/datasets/andrewt28/MolLM-dataset/tree/main).

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
@article{10.1093/bioinformatics/btae260,
    author = {Tang, Xiangru and Tran, Andrew and Tan, Jeffrey and Gerstein, Mark B},
    title = "{MolLM: a unified language model for integrating biomedical text with 2D and 3D molecular representations}",
    journal = {Bioinformatics},
    volume = {40},
    number = {Supplement_1},
    pages = {i357-i368},
    year = {2024},
    month = {06},
    abstract = "{The current paradigm of deep learning models for the joint representation of molecules and text primarily relies on 1D or 2D molecular formats, neglecting significant 3D structural information that offers valuable physical insight. This narrow focus inhibits the models’ versatility and adaptability across a wide range of modalities. Conversely, the limited research focusing on explicit 3D representation tends to overlook textual data within the biomedical domain.We present a unified pre-trained language model, MolLM, that concurrently captures 2D and 3D molecular information alongside biomedical text. MolLM consists of a text Transformer encoder and a molecular Transformer encoder, designed to encode both 2D and 3D molecular structures. To support MolLM’s self-supervised pre-training, we constructed 160K molecule-text pairings. Employing contrastive learning as a supervisory signal for learning, MolLM demonstrates robust molecular representation capabilities across four downstream tasks, including cross-modal molecule and text matching, property prediction, captioning, and text-prompted molecular editing. Through ablation, we demonstrate that the inclusion of explicit 3D representations improves performance in these downstream tasks.Our code, data, pre-trained model weights, and examples of using our model are all available at https://github.com/gersteinlab/MolLM. In particular, we provide Jupyter Notebooks offering step-by-step guidance on how to use MolLM to extract embeddings for both molecules and text.}",
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae260},
    url = {https://doi.org/10.1093/bioinformatics/btae260},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/40/Supplement\_1/i357/58355106/btae260.pdf},
}
```

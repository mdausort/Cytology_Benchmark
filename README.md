# A Comprehensive Benchmark of Foundation Models Fine-Tuning for Cytological Image Classification

**Currently being updated**

This repository provides a unified benchmark for fine-tuning foundation models on cytological image classification tasks.

It integrates and extends several existing open-source repositories, including prompt learning and adaptation frameworks such as CoOp, KgCoOp, PromptSRC, TaskRes, Tip-Adapter, and related projects. These submodules were originally developed in separate public GitHub repositories and were adapted here to support additional foundation models, cytology-specific datasets, and a unified experimental pipeline.

The main goal of this repository is to provide a common framework for comparing different adaptation strategies across multiple cytological image classification benchmarks.

## Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Original Repositories](#original-repositories)
- [Contact](#contact)

## Repository Structure

```bash
Cytology_Benchmark/
├── CoOp/
├── Cytology-fine-tuning/
├── KgCoOp/
├── multimodal-prompt-learning/
├── Prompt-align/
├── TaskRes/
├── Tip-Adapter/
├── DATASETS.md
├── README.md
└── LICENSE

## Installation 

📌 **NB:** The Python version used is 3.9.13.

1. Create a virtual environment
   This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the dassl environment first. Simply follow the instructions described here to install dassl as well as PyTorch.

2. Activate the environment
   ```bash
   source /env/dassl/bin/activate
   ```

3. Clone the GitHub repository
   ```bash
   pip3 install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
   git clone https://github.com/mdausort/Cytology_Benchmark.git
   ```
   After that, run pip install -r requirements.txt under CoOp/ to install a few more packages required by CLIP.
  
4. Install the required packages
   ```bash
   cd Cytology-fine-tuning
   pip3 install -r requirements.txt
   ```


3. Datasets downloads:

| Dataset        | 🔗 Download Link                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| APACC          | [📥 Link](https://osf.io/fp2xe/overview)                                                                |
| BCFC           | [📥 Link](https://www.kaggle.com/datasets/cmacus/body-cavity-fluid-cytology-images)                     |
| BloodMNIST     | [📥 Link](https://zenodo.org/records/10519652)                                                          |
| BMCD           | [📥 Link](https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification)                    |
| BMT            | [📥 Link](https://www.synapse.org/Synapse:syn55262661)                                                  |
| FNAC           | [📥 Link](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbC1UNmQtX0VOZjZheHNFYnZoYkVjMmdVRnM&cid=FAD710BFDFE9935F&id=FAD710BFDFE9935F%21107&parId=FAD710BFDFE9935F%21sea8cc6beffdb43d7976fbc7da445c639&o=OneUp) |
| Herlev         | [📥 Link](https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset)                          |
| HiCervix       | [📥 Link](https://zenodo.org/records/11087263)                                                          |
| MLCC           | [📥 Link](https://www.kaggle.com/datasets/blank1508/mendeley-lbc-cervical-cancer-)                      |
| SIPaKMeD       | [📥 Link](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed) |

# A comprehensive benchmark for adapting foundation models to cytological image classification under few-shot settings.

📄 Paper: [Submitted to JMI]

## Overview

This repository accompanies our work on benchmarking foundation models for cytological image classification in low-data regimes.

Cytology datasets are typically small and require expert annotations, making them ideal candidates for few-shot learning approaches. In this project, we evaluate multiple foundation models and parameter-efficient fine-tuning (PEFT) strategies across a diverse set of cytology datasets.

We compare:
- Vision Transformers (ViTs) and Vision-Language Models (VLMs)
- Different pretraining domains (natural, biomedical, histopathology)
- Several PEFT methods (LoRA, VPT, prompt learning, adapters)

All experiments are conducted in a few-shot setting (1 to 16 samples per class).

## Key findings

- LoRA consistently outperforms other PEFT methods for adapting foundation models
- Larger backbones improve performance, especially in extreme low-shot regimes
- Histopathology-pretrained models perform better in low-shot settings
- General-purpose models (e.g., CLIP) become competitive as more data is available
- Simple ensembling improves robustness and accuracy

## Repository structure

This repository integrates several existing frameworks, adapted to support new backbones and cytology datasets.

- `CoOp/`
- `KgCoOp/`
- `TaskRes/`
- `Tip-Adapter/`
- `Prompt-align/`
- `multimodal-prompt-learning/`
- `Cytology-fine-tuning/`

Each submodule originates from a different repository and has been adapted for:
- additional foundation models
- unified dataset handling
- consistent few-shot evaluation protocols

| Github                     | 🔗 Link                                                                           |
| -------------------------- | --------------------------------------------------------------------------------- |
| CoOp                       | [📥 Link](https://github.com/kaiyangzhou/coop)                                    |
| KgCoOp                     | [📥 Link](https://github.com/htyao89/KgCoOp)                                      |
| TaskRes                    | [📥 Link](https://github.com/geekyutao/TaskRes)                                   |
| Tip-Adapter                | [📥 Link](https://github.com/gaopengcuhk/Tip-Adapter)                             |
| Prompt-align               | [📥 Link](https://github.com/BeierZhu/Prompt-align)                               |
| multimodal-prompt-learning | [📥 Link](https://github.com/muzairkhattak/multimodal-prompt-learning)            |
| Cytology-fine-tuning       | [📥 Link](https://github.com/mdausort/Cytology-fine-tuning)                       |

## Environments
Two Python environments are used:

### 1. dassl
Base environment for most experiments.

### 2. dassl_prograd
Extended version including additional prompt-learning methods (e.g., ProGrad).

Both environments have been modified and require the provided `requirements.txt`.

We recommend using conda:

```bash
conda create -n cytology python=3.10
conda activate cytology
pip install -r requirements.txt
```

## Datasets

We evaluate on 10 public cytological datasets covering multiple organs and classification tasks.

See `DATASETS.md` for:
- download links
- preprocessing details
- dataset structure

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

## Running experiments

Experiments are launched through bash scripts.

These scripts define the full experimental configuration, including:
- dataset
- model / backbone
- number of shots
- seed
- training hyperparameters
- output paths

They are designed to be easily adapted to new settings and can also be used with Slurm array jobs for large-scale runs.

Example:

```bash
bash scripts/launch_run.sh or scripts/main_ivlp.sh
```

## Supported methods

- Linear probing
- LoRA (Low Rank Adaptation)
- CoOp / CoCoOp / KgCoOp / ProGrad
- Tip-Adapter / TaskRes
- VPT (Visual Prompt Tuning)
- IVLP (Independant Visual Language Prompting)

All methods are adapted to work with multiple backbones as BiomedCLIP, PLIP, PubMedCLIP, QUILT and CONCH for VLM or DinoBLOOM and UNI for ViT.

## Contact 

If you have any questions, you can contact us by email: [manon.dausort@uclouvain.be](mailto\:manon.dausort@uclouvain.be)

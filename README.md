# A Comprehensive Benchmark of Foundation Models Fine-Tuning for Cytological Image Classification

**Currently being updated**

## Contents 

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation 

📌 **NB:** The Python version used is 3.9.13.

1. Create a virtual environment
   ```bash
   python3 -m venv cyto_ft_venv
   source cyto_ft_venv/bin/activate
   ```

   Clone the GitHub repository
   ```bash
   pip3 install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
   git clone https://github.com/mdausort/Cytology-fine-tuning.git
   ```
   
   Install the required packages
   ```bash
   cd Cytology-fine-tuning
   pip3 install -r requirements.txt
   ```


2. Datasets downloads:

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


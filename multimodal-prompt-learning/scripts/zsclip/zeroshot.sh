#!/bin/bash
#SBATCH --job-name=zero_shot_eurosat
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/zero_shot_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/zero_shot_eurosat_%j.err"
#SBATCH --account=coalap

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning

# custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data
TRAINER=ZeroshotCLIP
DATASET=eurosat
CFG=vit_b32  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only
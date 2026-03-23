#!/bin/bash
#SBATCH --job-name=features_extractor_eurosat
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/features_extractor_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/features_extractor_eurosat_%j.err"
#SBATCH --account=coalap

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/CoOp/lpclip/

DATA=/gpfs/home/acad/ucl-elen/mdausort/data/
OUTPUT='./clip_feat/'
SEED=1

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for SPLIT in train val test
do
    python feat_extractor.py \
    --split ${SPLIT} \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file ../configs/datasets/eurosat.yaml \
    --config-file ../configs/trainers/CoOp/rn50_val.yaml \
    --output-dir ${OUTPUT} \
    --eval-only
done

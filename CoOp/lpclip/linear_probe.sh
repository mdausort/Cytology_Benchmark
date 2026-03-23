#!/bin/bash
#SBATCH --job-name=linear_probe_eurosat
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/linear_probe_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/linear_probe_eurosat_%j.err"
#SBATCH --account=coalap

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/CoOp/lpclip/

feature_dir=clip_feat

python linear_probe.py \
    --dataset EuroSAT \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
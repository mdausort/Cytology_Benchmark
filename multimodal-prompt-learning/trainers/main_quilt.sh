#!/bin/bash
#SBATCH --job-name=tpt
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning/trainers/quilt_debug.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning/trainers/quilt_debug.err"
#SBATCH --account=coalap

set -euo pipefail

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set -euo pipefail

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to the repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning/trainers

python -u main_quilt.py

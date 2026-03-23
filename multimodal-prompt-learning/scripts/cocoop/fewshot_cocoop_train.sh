#!/bin/bash
#SBATCH --job-name=coop_maple_eurosat
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/coop_maple_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/coop_maple_eurosat_%j.err"
#SBATCH --account=coalap

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning

# custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data
TRAINER=CoCoOp

DATASET=eurosat
CFG=vit_b32
SHOTS=1

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
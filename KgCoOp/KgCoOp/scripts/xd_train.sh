#!/bin/bash
#SBATCH --job-name=kgcoop_eurosat
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/kgcoop_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/kgcoop_eurosat_%j.err"
#SBATCH --account=coalap

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/KgCoOp/KgCoOp/

# custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data/
TRAINER=KgCoOp
WEIGHT=8.0
CFG=vit_b16_ep100_ctxv1 
CTP=end    # class token position (end or middle)
NCTX=4     # number of context tokens
SHOTS=16   # number of shots 
CSC=False  # class-specific context (False or True)


for DATASET in eurosat
do
for SEED in 1 2 3
do
    DIR=output_1120_xd/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
done


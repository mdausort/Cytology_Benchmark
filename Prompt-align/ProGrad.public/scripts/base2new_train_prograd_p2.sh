#!/bin/bash
#SBATCH --job-name=prograd_eurosat
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs_debug/prograd_eurosat_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs_debug/prograd_eurosat_%j.err"
#SBATCH --account=coalap
#SBATCH --array=0-199%100

export CUDA_VISIBLE_DEVICES=0

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl_prograd/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/Prompt-align/ProGrad.public/

# custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data/
TRAINER=ProGrad
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CSC=False  # class-specific context (False or True)
LAMBDA=1.0

DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 hicervix herlev mlcc sipakmed)
CFGS=(conch_b16_ep100 vit_b32_ep100 vit_b16_ep100 biomedclip_b16_ep100 quilt_b16_ep100 quilt_b32_ep100 pubmedclip_b32_ep100 plip_b32_ep100)
SHOTS_LIST=(1 2 4 8 16)
SEEDS=(1 2 3)

# DATASETS=(bcfc sipakmed)
# CFGS=(conch_b16_ep100 vit_b32_ep100 vit_b16_ep100 biomedclip_b16_ep100 quilt_b16_ep100 quilt_b32_ep100 pubmedclip_b32_ep100 plip_b32_ep100)  # conch_b16_ep100 vit_b32_ep100 vit_b16_ep100 biomedclip_b16_ep100 quilt_b16_ep100 quilt_b32_ep100 
# SHOTS_LIST=(1)
# SEEDS=(1)

# --- compute combination from SLURM_ARRAY_TASK_ID ---
nd=${#DATASETS[@]}
nc=${#CFGS[@]}
ns=${#SHOTS_LIST[@]}
nse=${#SEEDS[@]}

total=$((nd * nc * ns * nse))

OFFSET=1000
tid=$((SLURM_ARRAY_TASK_ID + OFFSET))

if (( tid < 0 || tid >= total )); then
  echo "Error: SLURM_ARRAY_TASK_ID=$tid out of range [0, $((total-1))]"
  exit 1
fi

# order: dataset -> cfg -> shots -> seed  (same nesting as your for-loops)
seed_idx=$(( tid % nse ))
tmp=$(( tid / nse ))

shots_idx=$(( tmp % ns ))
tmp=$(( tmp / ns ))

cfg_idx=$(( tmp % nc ))
dataset_idx=$(( tmp / nc ))

DATASET=${DATASETS[$dataset_idx]}
CFG=${CFGS[$cfg_idx]}
SHOTS=${SHOTS_LIST[$shots_idx]}
SEED=${SEEDS[$seed_idx]}

DIR=/gpfs/projects/acad/coalap/mdausort/ISBI_sup/output/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
LOGFILE="${DIR}/log.txt"

echo "Task ${tid}/${total}: DATASET=${DATASET} CFG=${CFG} SHOTS=${SHOTS} SEED=${SEED}"
echo "Output dir: ${DIR}"

# --- skip if already done ---
if [[ -f "${LOGFILE}" ]]; then
  echo "SKIP: log exists -> ${LOGFILE}"
  exit 0
fi

mkdir -p "${DIR}"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    LOSS.LAMBDA ${LAMBDA} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all

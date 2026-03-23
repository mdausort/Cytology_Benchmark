#!/bin/bash
#SBATCH --job-name=maple
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logsde/maple_%A_%a.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logsde/maple_%A_%a.err"
#SBATCH --account=coalap
#SBATCH --array=0-7%100

set -euo pipefail

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to the repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning

# IMPORTANT: forcer le code local si besoin
export PYTHONPATH="$PWD:$PWD/Dassl.pytorch:$PYTHONPATH"

DATA=/gpfs/home/acad/ucl-elen/mdausort/data/
TRAINER=MaPLe

DATASETS=(bcfc)  #  bloodmnist bmcd bmt eurosat fnac2019 herlev hicervix mlcc sipakmed
CFGS=(conch_b16_c2_batch4_2ctx biomedclip_b16_c2_batch4_2ctx quilt_b16_c2_batch4_2ctx quilt_b32_c2_batch4_2ctx plip_b32_c2_batch4_2ctx pubmedclip_b32_c2_batch4_2ctx vit_b16_c2_batch4_2ctx vit_b32_c2_batch4_2ctx)
SHOTS_LIST=(1)
SEEDS=(1)

# --- compute combination from SLURM_ARRAY_TASK_ID ---
nd=${#DATASETS[@]}
nc=${#CFGS[@]}
ns=${#SHOTS_LIST[@]}
nse=${#SEEDS[@]}

total=$((nd * nc * ns * nse))

OFFSET=0
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

DIR=/gpfs/projects/acad/coalap/mdausort/ISBI_sup/outputdebu/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all

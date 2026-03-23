#!/bin/bash
#SBATCH --job-name=both
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs_ivlp/ivlp_%A_%a.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs_ivlp/ivlp_%A_%a.err"
#SBATCH --account=coalap
#SBATCH --array=0-999%100

set -euo pipefail

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to the repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning


# IMPORTANT: forcer le code local si besoin
export PYTHONPATH="$PWD:$PWD/Dassl.pytorch:$PYTHONPATH"

# 3) custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data/
TRAINER=IVLP

# 4) grids (arrays)
DATASETS=(apacc bcfc bloodmnist bmcd bmt herlev fnac2019 hicervix sipakmed mlcc)
CFGS=(vit_b16 vit_b32 biomedclip_b16 conch_b16 quilt_b16 quilt_b32 pubmedclip_b32 plip_b32)
SHOTS_LIST=(1 2 4 8 16)
SEEDS=(1 2 3)

# 5) compute combination from SLURM_ARRAY_TASK_ID
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

# 6) order: dataset -> cfg -> shots -> seed  (same nesting as your for-loops)
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

PROMPT_DEPTH_TEXT=12
PROMPT_DEPTH_VISION=12
N_CTX_VISION=4
N_CTX_TEXT=4

DIR=/gpfs/projects/acad/coalap/mdausort/ISBI_sup/output/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/${N_CTX_VISION}nctxv_dv${PROMPT_DEPTH_VISION}_${N_CTX_TEXT}nctxt_dt${PROMPT_DEPTH_TEXT}/seed${SEED}
LOGFILE="${DIR}/log.txt"

echo "Task ${tid}/${total}: DATASET=${DATASET} CFG=${CFG} SHOTS=${SHOTS} SEED=${SEED}"
echo "Output dir: ${DIR}"

# 6) skip if already done
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
  TRAINER.IVLP.N_CTX_VISION ${N_CTX_VISION} \
  TRAINER.IVLP.N_CTX_TEXT ${N_CTX_TEXT} \
  TRAINER.IVLP.PROMPT_DEPTH_VISION ${PROMPT_DEPTH_VISION} \
  TRAINER.IVLP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH_TEXT} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES all

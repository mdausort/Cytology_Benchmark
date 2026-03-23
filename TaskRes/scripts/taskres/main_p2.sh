#!/bin/bash
#SBATCH --job-name=taskres
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/taskres_%j.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/taskres_%j.err"
#SBATCH --account=coalap
#SBATCH --array=0-199%100

# 1) activate your venv
source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate

# 2) go to CoOp repo
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/TaskRes/

# custom config
DATA=/gpfs/home/acad/ucl-elen/mdausort/data
TRAINER=TaskRes
ENHANCE=none                          # path to enhanced base weights
SCALE=0.5     

DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 hicervix herlev mlcc sipakmed)  # 
CFGS=(conch_b16 vit_b16 vit_b32 biomedclip_b16 quilt_b16 quilt_b32 pubmedclip_b32 plip_b32)
SHOTS_LIST=(1 2 4 8 16)
SEEDS=(1 2 3)

# DATASETS=(bloodmnist)
# CFGS=(vit_b16 biomedclip_b16 quilt_b16)
# SHOTS_LIST=(16)
# SEEDS=(1 2 3)

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

DIR=/gpfs/projects/acad/coalap/mdausort/ISBI_sup/output/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/seed${SEED}
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
    --enhanced-base ${ENHANCE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE}

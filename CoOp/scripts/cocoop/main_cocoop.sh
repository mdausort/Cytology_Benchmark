# --- activate your venv ---
source /env/dassl/bin/activate

# --- go to the repo ---
cd ./Cytology_Benchmark/CoOp

DATA=/path/to/datasets
TRAINER=CoCoOp

# --- grids (arrays) ---
DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 hicervix herlev mlcc sipakmed)
CFGS=(conch_vit_b16 clip_vit_b16 clip_vit_b32 biomedclip_vit_b16 quilt_vit_b16 quilt_vit_b32 plip_vit_b32 pubmedclip_vit_b32)
SHOTS_LIST=(1 2 4 8 16)
SEEDS=(1 2 3)

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

DIR=./Cytology_Benchmark/output/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/seed${SEED}
LOGFILE="${DIR}/log.txt"

echo "Task ${tid}/${total}: DATASET=${DATASET} CFG=${CFG} SHOTS=${SHOTS} SEED=${SEED}"
echo "Output dir: ${DIR}"

# --- skip if already done ---
if [[ -f "${LOGFILE}" ]]; then
  echo "SKIP: log exists -> ${LOGFILE}"
  exit 0
fi

mkdir -p "${DIR}"

# --- launch code ---
python train.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir ${DIR} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES all

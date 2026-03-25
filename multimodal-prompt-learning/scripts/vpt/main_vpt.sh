# --- activate your venv ---
source /env/dassl/bin/activate

# --- go to the repo ---
cd ./Cytology_Benchmark/multimodal-prompt-learning

DATA=/path/to/datasets
TRAINER=IVLP

# --- grids (arrays) ---
DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 herlev hicervix mlcc sipakmed)
CFGS=(4 8 16)
SHOTS_LIST=(1 2 4 8 16)
SEEDS=(1 2 3)
MODEL=clip_vit_b16  # Can be clip_vit_b16 clip_vit_b32 dinobloom_vit_s14 dinobloom_vit_b14 dinobloom_vit_l14 dinobloom_vit_g14

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
N_CTX_VISION=${CFGS[$cfg_idx]}
SHOTS=${SHOTS_LIST[$shots_idx]}
SEED=${SEEDS[$seed_idx]}

PROMPT_DEPTH_TEXT=0
PROMPT_DEPTH_VISION=24
N_CTX_TEXT=4

DIR=./Cytology_Benchmark/output/${DATASET}/${TRAINER}/${MODEL}/${SHOTS}shots/${N_CTX_VISION}nctxv_dv${PROMPT_DEPTH_VISION}/seed${SEED}
LOGFILE="${DIR}/log.txt"

echo "Task ${tid}/${total}: DATASET=${DATASET} CFG=${MODEL} SHOTS=${SHOTS} SEED=${SEED}"
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
  --config-file configs/trainers/${TRAINER}/${MODEL}.yaml \
  --output-dir ${DIR} \
  TRAINER.IVLP.N_CTX_VISION ${N_CTX_VISION} \
  TRAINER.IVLP.N_CTX_TEXT ${N_CTX_TEXT} \
  TRAINER.IVLP.PROMPT_DEPTH_VISION ${PROMPT_DEPTH_VISION} \
  TRAINER.IVLP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH_TEXT} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.IVLP.MODE vision-only

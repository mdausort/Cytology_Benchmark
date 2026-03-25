# --- activate your venv ---
source /env/dassl/bin/activate

# --- go to the repo ---
cd ./Cytology_Benchmark/multimodal-prompt-learning

DATA=/path/to/datasets
TRAINER=IVLP

# --- grids (arrays) ---
DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 herlev hicervix mlcc sipakmed)
CFGS=(clip_vit_b16 clip_vit_b32 biomedclip_vit_b16 conch_vit_b16 quilt_vit_b16 quilt_vit_b32 pubmedclip_vit_b32 plip_vit_b32)
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

PROMPT_DEPTH_VISION=12        # Number of vision layers where visual prompts are injected (set to 0 to disable visual prompting, i.e. TPT-style)
N_CTX_VISION=16               # Number of learnable visual tokens

PROMPT_DEPTH_TEXT=12          # Number of text layers where text prompts are injected (set to 0 to disable text prompting, i.e. VPT-style)
N_CTX_TEXT=16                 # Number of learnable text tokens

# Examples VPT then TPT
# PROMPT_DEPTH_VISION=x        # Learnable token injected inside x first layers
# N_CTX_VISION=y               # y learnable visual tokens
# PROMPT_DEPTH_TEXT=0
# N_CTX_TEXT=0

# PROMPT_DEPTH_VISION=0
# N_CTX_VISION=0
# PROMPT_DEPTH_TEXT=x          # Learnable token injected inside x first layers
# N_CTX_TEXT=y                 # y learnable text tokens

DIR=./Cytology_Benchmark/output/${DATASET}/${TRAINER}/${CFG}/${SHOTS}shots/${N_CTX_VISION}nctxv_dv${PROMPT_DEPTH_VISION}_${N_CTX_TEXT}nctxt_dt${PROMPT_DEPTH_TEXT}/seed${SEED}
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
  TRAINER.IVLP.N_CTX_VISION ${N_CTX_VISION} \
  TRAINER.IVLP.N_CTX_TEXT ${N_CTX_TEXT} \
  TRAINER.IVLP.PROMPT_DEPTH_VISION ${PROMPT_DEPTH_VISION} \
  TRAINER.IVLP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH_TEXT} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.IVLP.MODE ivlp

# --- activate your venv ---
source /env/dassl/bin/activate

# --- go to the repo ---
cd ./Cytology_Benchmark/Cytology-fine-tuning/

# --- grids (arrays) ---
SEEDS=(1 2 3)
SHOTS=(1 2 4 8 16)
DATASETS=(apacc bcfc bloodmnist bmcd bmt fnac2019 herlev hicervix mlcc sipakmed)
MODEL_NAMES=(clip-b16 clip-l14 dinobloom-s dinobloom-g dinobloom-l)

# --- compute combination from SLURM_ARRAY_TASK_ID ---
ns=${#SEEDS[@]}
nsh=${#SHOTS[@]}
nd=${#DATASETS[@]}
nb=${#MODEL_NAMES[@]}

TOTAL=$((ns * nsh * nd * nb))

OFFSET=0
TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))


if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
    echo "Invalid task id ${TASK_ID} (TOTAL=${TOTAL})"
    exit 1
fi

# order: dataset -> cfg -> shots -> seed  (same nesting as your for-loops)
seed_idx=$(( TASK_ID % ns ))
tmp=$(( TASK_ID / ns ))

shots_idx=$(( tmp % nsh ))
tmp=$(( tmp / nsh ))

dataset_idx=$(( tmp % nd ))
model_name_idx=$(( tmp / nd ))

SEED=${SEEDS[$seed_idx]}
SHOT=${SHOTS[$shots_idx]}
DATASET=${DATASETS[$dataset_idx]}
MODEL_NAME=${MODEL_NAMES[$model_name_idx]}

declare -A BACKBONES=(
    [clip-b16]="ViT-B/16"
    [clip-b32]="ViT-B/32"
    [clip-l14]="ViT-L/14"
    [pubmedclip]="ViT-B/32"
    [conch]="ViT-B/16"
    [plip]="ViT-B/32"
    [quilt-b16]="ViT-B/16"
    [quilt-b32]="ViT-B/32"
    [biomedclip]="ViT-B/16"
    [dinobloom-s]="ViT-S/14"
    [dinobloom-b]="ViT-B/14"
    [dinobloom-l]="ViT-L/14"
    [dinobloom-g]="ViT-G/14"
    [uni]="ViT-L/14"
    [vit_google-b16]="ViT-B/16"
    [vit_google-b32]="ViT-B/32"
)

BACKBONE="${BACKBONES[$MODEL_NAME]:-}"
if [[ -z "$BACKBONE" ]]; then
    echo "Unknown MODEL_NAME=${MODEL_NAME} for backbone selection"
    exit 1
fi

BACKBONE="${BACKBONES[$MODEL_NAME]:-}"
if [[ -z "$BACKBONE" ]]; then
    echo "Unknown MODEL_NAME=${MODEL_NAME} for backbone selection"
    exit 1
fi

echo "Running: model_name=${MODEL_NAME} backbone=${BACKBONE}, dataset=${DATASET}, shots=${SHOT}, seed=${SEED}"

RANK=2
ENCODER="vision"

# --- launch code ---
python3 run.py \
  --log_path ./Cytology_Benchmark/output/${DATASET}/CLIP-LoRA/${ENCODER}_vlm/${MODEL_NAME}/rank${RANK}/${SHOT}shots/seed${SEED} \
  --seed ${SEED} \
  --shots ${SHOT} \
  --lr 0.001 \
  --n_epochs 100 \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET} \
  --task lora-vlm \
  --encoder ${ENCODER} \
  --backbone ${BACKBONE} \
  --r ${RANK} \
  --results_path ./Cytology_Benchmark/output/${DATASET}/CLIP-LoRA/${ENCODER}_vlm/${MODEL_NAME}/rank${RANK}/${SHOT}shots/seed${SEED}

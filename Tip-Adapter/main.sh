# --- activate your venv ---
source /env/dassl/bin/activate

# --- go to the repo ---
cd ./Cytology_Benchmark/Tip-Adapter/

TRAINER="Tip-Adapter"

SEEDS=(1 2 3)
SHOTS=(1 2 4 8 16)

CFGS=(
    configs/apacc_clip_vit_b16.yaml
    configs/apacc_clip_vit_b32.yaml
    configs/apacc_quilt_vit_b16.yaml
    configs/apacc_quilt_vit_b32.yaml
    configs/apacc_biomedclip_vit_b16.yaml
    configs/apacc_conch_vit_b16.yaml
    configs/apacc_pubmedclip_vit_b32.yaml
    configs/apacc_plip_vit_b32.yaml

    configs/bcfc_clip_vit_b16.yaml
    configs/bcfc_clip_vit_b32.yaml
    configs/bcfc_quilt_vit_b16.yaml
    configs/bcfc_quilt_vit_b32.yaml
    configs/bcfc_biomedclip_vit_b16.yaml
    configs/bcfc_conch_vit_b16.yaml
    configs/bcfc_pubmedclip_vit_b32.yaml
    configs/bcfc_plip_vit_b32.yaml
    
    configs/bloodmnist_clip_vit_b16.yaml
    configs/bloodmnist_clip_vit_b32.yaml
    configs/bloodmnist_quilt_vit_b16.yaml
    configs/bloodmnist_quilt_vit_b32.yaml
    configs/bloodmnist_biomedclip_vit_b16.yaml
    configs/bloodmnist_conch_vit_b16.yaml
    configs/bloodmnist_pubmedclip_vit_b32.yaml
    configs/bloodmnist_plip_vit_b32.yaml
    
    configs/bmt_clip_vit_b16.yaml
    configs/bmt_clip_vit_b32.yaml
    configs/bmt_quilt_vit_b16.yaml
    configs/bmt_quilt_vit_b32.yaml
    configs/bmt_biomedclip_vit_b16.yaml
    configs/bmt_conch_vit_b16.yaml
    configs/bmt_pubmedclip_vit_b32.yaml
    configs/bmt_plip_vit_b32.yaml
    
    configs/bmcd_clip_vit_b16.yaml
    configs/bmcd_clip_vit_b32.yaml
    configs/bmcd_quilt_vit_b16.yaml
    configs/bmcd_quilt_vit_b32.yaml
    configs/bmcd_biomedclip_vit_b16.yaml
    configs/bmcd_conch_vit_b16.yaml
    configs/bmcd_pubmedclip_vit_b32.yaml
    configs/bmcd_plip_vit_b32.yaml
    
    configs/fnac2019_clip_vit_b16.yaml
    configs/fnac2019_clip_vit_b32.yaml
    configs/fnac2019_quilt_vit_b16.yaml
    configs/fnac2019_quilt_vit_b32.yaml
    configs/fnac2019_biomedclip_vit_b16.yaml
    configs/fnac2019_conch_vit_b16.yaml
    configs/fnac2019_pubmedclip_vit_b32.yaml
    configs/fnac2019_plip_vit_b32.yaml
    
    configs/herlev_clip_vit_b16.yaml
    configs/herlev_clip_vit_b32.yaml
    configs/herlev_quilt_vit_b16.yaml
    configs/herlev_quilt_vit_b32.yaml
    configs/herlev_biomedclip_vit_b16.yaml
    configs/herlev_conch_vit_b16.yaml
    configs/herlev_pubmedclip_vit_b32.yaml
    configs/herlev_plip_vit_b32.yaml
    
    configs/hicervix_clip_vit_b16.yaml
    configs/hicervix_clip_vit_b32.yaml
    configs/hicervix_quilt_vit_b16.yaml
    configs/hicervix_quilt_vit_b32.yaml
    configs/hicervix_biomedclip_vit_b16.yaml
    configs/hicervix_conch_vit_b16.yaml
    configs/hicervix_pubmedclip_vit_b32.yaml
    configs/hicervix_plip_vit_b32.yaml
    
    configs/mlcc_clip_vit_b16.yaml
    configs/mlcc_clip_vit_b32.yaml
    configs/mlcc_quilt_vit_b16.yaml
    configs/mlcc_quilt_vit_b32.yaml
    configs/mlcc_biomedclip_vit_b16.yaml
    configs/mlcc_conch_vit_b16.yaml
    configs/mlcc_pubmedclip_vit_b32.yaml
    configs/mlcc_plip_vit_b32.yaml
    
    configs/sipakmed_clip_vit_b16.yaml
    configs/sipakmed_clip_vit_b32.yaml
    configs/sipakmed_quilt_vit_b16.yaml
    configs/sipakmed_quilt_vit_b32.yaml
    configs/sipakmed_biomedclip_vit_b16.yaml
    configs/sipakmed_conch_vit_b16.yaml
    configs/sipakmed_pubmedclip_vit_b32.yaml
    configs/sipakmed_plip_vit_b32.yaml
)

# --- compute combination from SLURM_ARRAY_TASK_ID ---
N_SEEDS=${#SEEDS[@]}
N_SHOTS=${#SHOTS[@]}
N_CFGS=${#CFGS[@]}
N_TASKS=$((N_CFGS * N_SHOTS * N_SEEDS))

OFFSET=0
tid=$((SLURM_ARRAY_TASK_ID + OFFSET))

if (( tid < 0 || tid >= N_TASKS )); then
  echo "Error: SLURM_ARRAY_TASK_ID=$tid out of range [0, $((N_TASKS-1))]"
  exit 1
fi

CFG_IDX=$(( tid / (N_SHOTS * N_SEEDS) ))
REM=$(( tid % (N_SHOTS * N_SEEDS) ))
SHOT_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

CFG="${CFGS[$CFG_IDX]}"
SHOT="${SHOTS[$SHOT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# --- launch code ---
python main.py \
  --config "${CFG}" \
  --shots "${SHOT}" \
  --seed "${SEED}"

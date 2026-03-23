#!/bin/bash
#SBATCH --job-name=tip_adapter
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --output="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/tip_adapter_%A_%a.out"
#SBATCH --error="/gpfs/projects/acad/coalap/mdausort/ISBI_sup/logs/tip_adapter_%A_%a.err"
#SBATCH --account=coalap
#SBATCH --array=0-999%100

set -euo pipefail

source /gpfs/home/acad/ucl-elen/mdausort/env/dassl/bin/activate
cd /gpfs/projects/acad/coalap/mdausort/ISBI_sup/Tip-Adapter/

TRAINER="Tip-Adapter"

SEEDS=(1 2 3)
SHOTS=(1 2 4 8 16)

CFGS=(
    configs/apacc_ViT16.yaml
    configs/apacc_ViT32.yaml
    configs/apacc_quiltViT16.yaml
    configs/apacc_quiltViT32.yaml
    configs/apacc_biomedclipViT16.yaml
    configs/apacc_conchViT16.yaml
    configs/apacc_pubmedclipViT32.yaml
    configs/apacc_plipViT32.yaml

    configs/bcfc_ViT16.yaml
    configs/bcfc_ViT32.yaml
    configs/bcfc_quiltViT16.yaml
    configs/bcfc_quiltViT32.yaml
    configs/bcfc_biomedclipViT16.yaml
    configs/bcfc_conchViT16.yaml
    configs/bcfc_pubmedclipViT32.yaml
    configs/bcfc_plipViT32.yaml
    
    configs/bloodmnist_ViT16.yaml
    configs/bloodmnist_ViT32.yaml
    configs/bloodmnist_quiltViT16.yaml
    configs/bloodmnist_quiltViT32.yaml
    configs/bloodmnist_biomedclipViT16.yaml
    configs/bloodmnist_conchViT16.yaml
    configs/bloodmnist_pubmedclipViT32.yaml
    configs/bloodmnist_plipViT32.yaml
    
    configs/bmt_ViT16.yaml
    configs/bmt_ViT32.yaml
    configs/bmt_quiltViT16.yaml
    configs/bmt_quiltViT32.yaml
    configs/bmt_biomedclipViT16.yaml
    configs/bmt_conchViT16.yaml
    configs/bmt_pubmedclipViT32.yaml
    configs/bmt_plipViT32.yaml
    
    configs/bmcd_ViT16.yaml
    configs/bmcd_ViT32.yaml
    configs/bmcd_quiltViT16.yaml
    configs/bmcd_quiltViT32.yaml
    configs/bmcd_biomedclipViT16.yaml
    configs/bmcd_conchViT16.yaml
    configs/bmcd_pubmedclipViT32.yaml
    configs/bmcd_plipViT32.yaml
    
    configs/fnac2019_ViT16.yaml
    configs/fnac2019_ViT32.yaml
    configs/fnac2019_quiltViT16.yaml
    configs/fnac2019_quiltViT32.yaml
    configs/fnac2019_biomedclipViT16.yaml
    configs/fnac2019_conchViT16.yaml
    configs/fnac2019_pubmedclipViT32.yaml
    configs/fnac2019_plipViT32.yaml
    
    configs/herlev_ViT16.yaml
    configs/herlev_ViT32.yaml
    configs/herlev_quiltViT16.yaml
    configs/herlev_quiltViT32.yaml
    configs/herlev_biomedclipViT16.yaml
    configs/herlev_conchViT16.yaml
    configs/herlev_pubmedclipViT32.yaml
    configs/herlev_plipViT32.yaml
    
    configs/hicervix_ViT16.yaml
    configs/hicervix_ViT32.yaml
    configs/hicervix_quiltViT16.yaml
    configs/hicervix_quiltViT32.yaml
    configs/hicervix_biomedclipViT16.yaml
    configs/hicervix_conchViT16.yaml
    configs/hicervix_pubmedclipViT32.yaml
    configs/hicervix_plipViT32.yaml
    
    configs/mlcc_ViT16.yaml
    configs/mlcc_ViT32.yaml
    configs/mlcc_quiltViT16.yaml
    configs/mlcc_quiltViT32.yaml
    configs/mlcc_biomedclipViT16.yaml
    configs/mlcc_conchViT16.yaml
    configs/mlcc_pubmedclipViT32.yaml
    configs/mlcc_plipViT32.yaml
    
    configs/sipakmed_ViT16.yaml
    configs/sipakmed_ViT32.yaml
    configs/sipakmed_quiltViT16.yaml
    configs/sipakmed_quiltViT32.yaml
    configs/sipakmed_biomedclipViT16.yaml
    configs/sipakmed_conchViT16.yaml
    configs/sipakmed_pubmedclipViT32.yaml
    configs/sipakmed_plipViT32.yaml
)

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

# Mapping: task -> (cfg, shot, seed)
CFG_IDX=$(( tid / (N_SHOTS * N_SEEDS) ))
REM=$(( tid % (N_SHOTS * N_SEEDS) ))
SHOT_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

CFG="${CFGS[$CFG_IDX]}"
SHOT="${SHOTS[$SHOT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

python main.py --config "${CFG}" --shots "${SHOT}" --seed "${SEED}"

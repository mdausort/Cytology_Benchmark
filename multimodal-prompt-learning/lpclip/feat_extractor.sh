# sh feat_extractor.sh
DATA=/gpfs/home/acad/ucl-elen/mdausort/data
OUTPUT='./clip_feat/'
SEED=1
DATASET=eurosat
# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for SPLIT in train val test
do
    python feat_extractor.py \
    --split ${SPLIT} \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning/configs/datasets/${DATASET}.yaml \
    --config-file /gpfs/projects/acad/coalap/mdausort/ISBI_sup/multimodal-prompt-learning/configs/trainers/CoOp/vit_b32.yaml \
    --output-dir ${OUTPUT} \
    --eval-only
done

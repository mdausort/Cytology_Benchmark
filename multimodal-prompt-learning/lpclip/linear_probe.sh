feature_dir=clip_feat

python linear_probe.py \
--dataset eurosat \
--feature_dir ${feature_dir} \
--num_step 8 \
--num_run 3


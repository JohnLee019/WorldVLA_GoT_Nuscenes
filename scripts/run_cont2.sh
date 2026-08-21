#!/bin/bash
cd ../VLA-GoT-release || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path ./output/nuscenes_trainval_full_r256/epoch0 --ft true \
  --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --data_config_train ./data/nuscenes_records/nuscenes_v1.0-trainval_train.json \
  --data_config_val_ind ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --data_config_val_ood ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --norm_path ./data/nuscenes_records/nuscenes_norm.json \
  --output_dir ./output/nuscenes_trainval_full_r256_cont2 \
  --trainable full --optimizer paged_adamw8bit \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --save_iteration_interval 2000 --ckpt_max_keep 2 \
  --epochs 2 --lr 2e-5 --precision bf16 --checkpointing

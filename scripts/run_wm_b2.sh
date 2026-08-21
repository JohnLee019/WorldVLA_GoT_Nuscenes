#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=3 train_nuscenes_wm.py \
  --resume_path ../ckpts/Lumina-mGPT-7B-768 --ft true \
  --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --norm_path ./data/nuscenes_records/nuscenes_norm.json \
  --data_config_train   ./data/nuscenes_wm_records/nuscenes_wm_v1.0-trainval_train.json \
  --data_config_val_ind ./data/nuscenes_wm_records/nuscenes_wm_v1.0-trainval_val.json \
  --data_config_val_ood ./data/nuscenes_wm_records/nuscenes_wm_val_slice200.json \
  --output_dir ./output/nuscenes_wm_trainval_full \
  --trainable full --optimizer paged_adamw8bit \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --save_iteration_interval 2000 --ckpt_max_keep 2 \
  --epochs 2 --lr 1e-4 --precision bf16 --checkpointing

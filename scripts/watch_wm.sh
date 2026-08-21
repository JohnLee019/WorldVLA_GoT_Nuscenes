#!/usr/bin/env bash
echo "== elapsed =="
ps -ewwo etime,args | grep -a '[e]val_wm_image_nuscenes' \
  | awk '{printf "  %-12s %s\n", $1, ($0 ~ /results\/wm_main/ ? "wm_main" : "wm_eps")}'
echo "== progress (one line per 10 records) =="
for f in results/wm_main.log results/wm_eps.log; do
  printf "  %-14s %s\n" "$(basename "$f")" "$(grep -aE '^  [0-9]+/' "$f" | tail -1)"
done
echo "== errors (should be empty) =="
grep -aE 'Traceback|CUDA out of memory' results/wm_main.log results/wm_eps.log | tail -3
echo "== gpu =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

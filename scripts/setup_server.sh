#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Set up a Linux GPU server for nuScenes trajectory-planning training.
#
# Run from inside the repo:
#     cd ~/VLA-GoT-release && bash setup_server.sh
#
# Layout this produces (ckpts MUST be the repo's sibling -- item_processor.py
# hardcodes ../ckpts/chameleon/tokenizer/, resolved against the run CWD):
#
#     ~/
#     ├── VLA-GoT-release/          <- run training from here
#     └── ckpts/
#         ├── chameleon/tokenizer/  <- YOU must upload this (see step 4)
#         ├── VLA_model_256/libero_spatial/
#         └── Lumina-mGPT-7B-768/
#
# Deliberately NOT installed: robosuite / mujoco / gym / bddl. Those are the
# LIBERO simulator stack; nuScenes driving never touches a simulator, and the
# only top-level import that used to drag them in (data/dataset.py) is now lazy.
#
# Deliberately NOT downloaded: the 14GB Lumina-mGPT safetensors. --tokenizer_path
# only ever reaches AutoTokenizer.from_pretrained(); the model weights come from
# --resume_path (VLA_model_256). ~10MB of tokenizer files is enough -- this was
# verified end-to-end locally, not assumed.

set -euo pipefail

ENV_NAME="${ENV_NAME:-got_vla}"
CUDA_TAG="${CUDA_TAG:-cu121}"     # match the server driver: cu118 / cu121 ...
PY_VER="3.10"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$(dirname "$REPO_DIR")/ckpts"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. sanity
say "0/6  sanity check"
[[ -f "$REPO_DIR/train_nuscenes.py" ]] || die "run this from inside VLA-GoT-release (train_nuscenes.py not found)"
command -v conda >/dev/null || die "conda not found on PATH"
nvidia-smi -L || die "no GPU visible. Training needs CUDA (the solver hardcodes the NCCL backend, so it is Linux+GPU only)."
echo "repo : $REPO_DIR"
echo "ckpts: $CKPT_DIR"

# ---------------------------------------------------------------- 1. conda env
say "1/6  conda env '$ENV_NAME' (python $PY_VER)"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env already exists, reusing"
else
    conda create -n "$ENV_NAME" python="$PY_VER" -y
fi
conda activate "$ENV_NAME"
echo "python: $(which python)"

# ---------------------------------------------------------------- 2. packages
say "2/6  installing packages"

# torch first so nothing else picks a different build
pip install "torch==2.2.0" "torchvision==0.17.0" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# training stack (no simulator)
pip install \
    "transformers==4.43.0" \
    "tokenizers==0.19.1" \
    "accelerate==0.33.0" \
    "safetensors==0.4.2" \
    "fairscale==0.4.13" \
    "sentencepiece==0.1.99" \
    "Pillow==10.2.0" \
    "einops==0.6.1" \
    "timm==1.0.3" \
    "h5py==3.14.0" \
    "tensorboard==2.17.1" \
    "tqdm==4.66.5" \
    "huggingface-hub==0.23.4" \
    pyyaml omegaconf

# nuScenes devkit -- pulls matplotlib/opencv/scikit-learn and, crucially, may
# drag numpy up to 2.x, which silently breaks torch 2.2.0 ("_ARRAY_API not found").
pip install nuscenes-devkit

# xllmx itself
pip install -e "$REPO_DIR"

# ---- pins, applied LAST and in this order (both have bitten this project) ----
# numpy: nuscenes-devkit / setup.py may have bumped it past 1.x
pip install "numpy==1.26.4"
# transformers: anything newer requires torch>=2.4 and quietly disables the
# PyTorch backend -> "cannot import name 'ChameleonForConditionalGeneration'".
# --no-deps so this cannot drag torch along with it.
pip install --force-reinstall --no-deps "transformers==4.43.0" "tokenizers==0.19.1"

# ---------------------------------------------------------------- 3. downloads
say "3/6  downloading model + tokenizer into $CKPT_DIR"
mkdir -p "$CKPT_DIR"
python - "$CKPT_DIR" <<'PY'
import sys, os
from huggingface_hub import snapshot_download

ckpt = sys.argv[1]

# the actual VLA weights we fine-tune from (~15GB)
snapshot_download(
    repo_id="Alibaba-DAMO-Academy/RynnVLA-002",
    allow_patterns="VLA_model_256/libero_spatial/*",
    local_dir=ckpt,
)
print("VLA_model_256 ok")

# tokenizer files ONLY -- not the 14GB safetensors (see header note)
snapshot_download(
    repo_id="Alpha-VLLM/Lumina-mGPT-7B-768",
    allow_patterns=["tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "config.json",
                    "generation_config.json"],
    local_dir=os.path.join(ckpt, "Lumina-mGPT-7B-768"),
)
print("Lumina tokenizer ok")
PY

# ---------------------------------------------------------------- 4. chameleon
say "4/6  checking chameleon VQGAN tokenizer (must be uploaded by hand)"
CHAM="$CKPT_DIR/chameleon/tokenizer"
missing=0
for f in text_tokenizer.json vqgan.yaml vqgan.ckpt; do
    if [[ -s "$CHAM/$f" ]]; then
        printf '  ok      %s (%s)\n' "$f" "$(du -h "$CHAM/$f" | cut -f1)"
    else
        printf '  MISSING %s\n' "$f"; missing=1
    fi
done
if [[ $missing -eq 1 ]]; then
    cat <<EOF

  These three files are NOT on HuggingFace -- not in RynnVLA-002 and not in
  Lumina-mGPT-7B-768 (both checked). They originate from Meta's Chameleon
  release, and this project's README never mentions them. Without them
  FlexARItemProcessor_*.__init__ dies with FileNotFoundError before training
  starts, so nothing below will work.

  Copy them from the local machine:
      scp -r <local>/ckpts/chameleon <user>@<server>:$CKPT_DIR/

  Expected: text_tokenizer.json ~5.4MB, vqgan.yaml ~1.5KB, vqgan.ckpt ~268MB
EOF
    die "chameleon tokenizer missing -- upload it, then re-run this script"
fi

# ---------------------------------------------------------------- 5. verify
say "5/6  verifying installation"
cd "$REPO_DIR"
python scripts/verify_setup.py

# ---------------------------------------------------------------- 6. next
say "6/6  done -- next steps"
cat <<EOF

  conda activate $ENV_NAME

  1) get nuScenes v1.0-mini onto this box, then re-run preprocessing HERE
     (the json stores ABSOLUTE image paths, so a json copied from another
      machine is useless):

     python -m data.preprocess_nuscenes \\
         --dataroot /path/to/nuscenes \\
         --version v1.0-mini \\
         --out_dir ./data/nuscenes_records

  2) smoke-test training (mini is ~344 records -- this only proves the loop
     runs, it will not learn anything meaningful):

     torchrun --nproc_per_node=1 train_nuscenes.py \\
         --resume_path $CKPT_DIR/VLA_model_256/libero_spatial \\
         --tokenizer_path $CKPT_DIR/Lumina-mGPT-7B-768 \\
         --data_config_train   ./data/nuscenes_records/nuscenes_v1.0-mini_train.json \\
         --data_config_val_ind ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \\
         --data_config_val_ood ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \\
         --norm_path ./data/nuscenes_records/nuscenes_norm.json \\
         --output_dir ./output/nuscenes_mini \\
         --trainable head --ft true \\
         --batch_size 1 --accum_iter 8 --epochs 20 --lr 1e-4 \\
         --precision bf16 --checkpointing

     --ft true is REQUIRED: the released ckpt has no optimizer state to resume.
     --batch_size 1 until the mini run is green: ActionHead.forward silently
     skips any batch row with <2 action markers while the GT side still counts
     that row, which would misalign the L1 loss. nuScenes always emits 6 markers
     per sample so B>1 should be safe, but that is untested. See train_nuscenes.py.

  3) evaluate:

     python eval_nuscenes.py \\
         --resume_path ./output/nuscenes_mini/epoch19 \\
         --tokenizer_path $CKPT_DIR/Lumina-mGPT-7B-768 \\
         --records_json ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \\
         --norm_path ./data/nuscenes_records/nuscenes_norm.json \\
         --output_dir ./results/nuscenes_eval

EOF

"""Pre-flight for the ego-status arm. Run this BEFORE the 33 h retrain.

Every check here fails in a way that a training run would NOT: the loss would go
down, the checkpoints would save, and the eval would produce a plausible-looking
L2. These are the failures that only show up as "ego status did not help".

  [A] the norm json actually carries driving state ranges
  [B] real driving states do not saturate the +-1 discretization range
  [C] the LIBERO-range guard fires when the norm json is missing them
  [D] ★train and eval tokenize the same record identically
  [E] invalid (scene-start) states are zeroed, and no record was dropped

[D] is the one worth the trouble. Training builds the conversation in
data/dataset_nuscenes.py and eval builds it in eval_nuscenes.build_planning_conv;
if they disagree on placeholder order, the model is trained on one layout and
evaluated on another with nothing raising.

Needs the Chameleon tokenizer assets (`../ckpts/.../tokenizer/`) because it
tokenizes for real. GPU is optional -- pass `--device cpu` if the card is busy.

Usage
-----
  python analysis/verify_state_plumbing.py \
      --records   ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
      --norm_path ./data/nuscenes_records/nuscenes_norm.json \
      --tokenizer ../ckpts/Lumina-mGPT-7B-768
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--norm_path", default="./data/nuscenes_records/nuscenes_norm.json")
    ap.add_argument("--tokenizer", default="../ckpts/Lumina-mGPT-7B-768")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200, help="records to sample for [B]")
    args = ap.parse_args()

    records = json.load(open(args.records))
    stats = json.load(open(args.norm_path))
    from data.preprocess_nuscenes import STATE_DIM, STATE_KEYS

    # ------------------------------------------------------------------ A
    print("\n[A] norm json carries driving state ranges")
    has = "state_min" in stats and "state_max" in stats
    check("state_min / state_max present", has,
          "rerun data/preprocess_nuscenes.py -- without these the item processor "
          "falls back to nothing and refuses to run" if not has else "")
    if not has:
        return finish()
    smin = np.array(stats["state_min"], float)
    smax = np.array(stats["state_max"], float)
    check(f"dims == STATE_DIM ({STATE_DIM})", len(smin) == STATE_DIM, f"got {len(smin)}")
    check("keys recorded", stats.get("state_keys") == STATE_KEYS, str(stats.get("state_keys")))
    check("ranges are non-degenerate", bool(np.all(smax > smin)),
          f"min={smin.tolist()} max={smax.tolist()}")
    # A driving v_x range that looks like a LIBERO end-effector range (|v| < 3)
    # is the exact symptom of the hardcoded-ranges bug leaking through.
    check("v_x range looks like a speed, not a robot arm", smax[0] > 5.0,
          f"v_x max = {smax[0]:.3f} m/s")

    # ------------------------------------------------------------------ E
    print("\n[E] state validity")
    with_state = [r for r in records if "state" in r]
    check("every record carries `state`", len(with_state) == len(records),
          f"{len(records) - len(with_state)} missing")
    invalid = [r for r in records if not r.get("state_valid", 1)]
    zeroed = all(all(abs(v) < 1e-9 for v in r["state"]) for r in invalid)
    check("invalid states are zeroed", zeroed, f"{len(invalid)} invalid records")
    print(f"       {len(invalid)}/{len(records)} scene-start records "
          f"({100*len(invalid)/max(1,len(records)):.2f}%) -- these are NOT dropped, "
          f"so the frozen 600-record eval set is unaffected")

    # ------------------------------------------------------------------ B/C
    print("\n[B] real states do not saturate the discretization range")
    from data.item_processor import FlexARItemProcessor_Action_NuScenes
    ip = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer, target_size=args.resolution,
        norm_path=args.norm_path, device=args.device)

    sample = [r for r in records if r.get("state_valid", 1)][: args.n]
    norm = np.array([ip.norm_state(r["state"]) for r in sample])
    sat = float(np.mean(np.abs(np.abs(norm) - 1.0) < 1e-6))
    check("saturation < 1% of state values", sat < 0.01, f"saturated fraction = {sat:.3%}")
    # If every record tokenizes to the same bins the channel carries zero
    # information, which is exactly what the LIBERO ranges would produce.
    toks = [tuple(ip.process_state(r["state"])["input_ids"]) for r in sample]
    check("states tokenize to many distinct patterns", len(set(toks)) > len(sample) // 2,
          f"{len(set(toks))} distinct / {len(sample)} records")
    for j, k in enumerate(STATE_KEYS):
        print(f"       {k:<9} normed mean {norm[:, j].mean():+7.4f}  "
              f"range [{norm[:, j].min():+7.4f}, {norm[:, j].max():+7.4f}]")

    print("\n[C] the missing-ranges guard fires")
    check("<|state|> is registered when ranges exist", "<|state|>" in ip.media_symbols,
          str(ip.media_symbols))
    ip_bare = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer, target_size=args.resolution,
        norm_path=None, device=args.device)
    # A norm json without state ranges must reproduce the pre-ego-status
    # processor exactly -- otherwise evaluating the incumbent checkpoint after
    # this change would tokenize differently than the run that produced 3.5557.
    check("<|state|> is NOT registered without ranges",
          "<|state|>" not in ip_bare.media_symbols, str(ip_bare.media_symbols))
    check("stateless media symbols unchanged",
          list(ip_bare.media_symbols) == ["<|image|>", "<|action|>"],
          str(ip_bare.media_symbols))
    try:
        ip_bare.norm_state(sample[0]["state"])
        check("norm_state refuses without driving ranges", False,
              "it silently used the inherited LIBERO ranges")
    except RuntimeError:
        check("norm_state refuses without driving ranges", True)

    # ------------------------------------------------------------------ D
    print("\n[D] train and eval tokenize the same record identically")
    from data.dataset_nuscenes import NuScenesFinetuneConversation, DEFAULT_PROMPT
    from eval_nuscenes import build_planning_conv

    from got_drive.eval_crop import crop_for_eval

    ds = NuScenesFinetuneConversation(args.records, resolution=args.resolution,
                                      with_state=True)
    idx = next(i for i, r in enumerate(records) if r.get("state_valid", 1))
    conv_t, imgs_t, act_t, sta_t = ds[idx]

    # ★Crop ONCE and hand the SAME cropped image to both paths.
    #
    # process_image() random-crops (sec.1.4: `center_crop` is misnamed), so two
    # independent loads of one frame tokenize to different image tokens no
    # matter what the state plumbing does -- an earlier version of this check
    # failed at token 20 for exactly that reason and said nothing about state.
    # On an already-crop-sized image var_center_crop is the identity, so this
    # makes the image region deterministic and leaves the check measuring what
    # it is for: placeholder ORDER and the state group.
    image = crop_for_eval(Image.open(records[idx]["images"][0]).convert("RGB"), ip, False)

    train_tokens, _ = ip.process_item(
        {"conversations": conv_t, "image": [image], "action": act_t, "state": sta_t},
        training_mode=True)
    eval_tokens = ip.process_item(
        build_planning_conv(DEFAULT_PROMPT, image, records[idx]["state"]),
        training_mode=False)

    # eval stops where the model must start generating, so its sequence is a
    # PREFIX of the training one (which continues into the action groups).
    n = len(eval_tokens)
    ok = len(train_tokens) >= n and list(train_tokens[:n]) == list(eval_tokens)
    detail = ""
    if not ok:
        first = next((i for i in range(min(n, len(train_tokens)))
                      if train_tokens[i] != eval_tokens[i]), min(n, len(train_tokens)))
        detail = (f"diverge at token {first}: train={train_tokens[first:first+6]} "
                  f"eval={eval_tokens[first:first+6]}")
    check("eval token sequence is a prefix of the training sequence", ok, detail)

    state_start = ip.token2id(ip.state_start_token)
    check("state tokens present in the training sequence", state_start in list(train_tokens))
    check("state tokens present in the eval sequence", state_start in list(eval_tokens))

    return finish()


def finish():
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{'='*66}\n{len(RESULTS)-n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("DO NOT START THE RETRAIN -- fix the failures above first.")
    else:
        print("plumbing is sound; the smoke test on one GPU is the next step.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

# WorldVLA-GoT — **A deliberation layer does not win** at driving planning

We put a **Graph-of-Thought-style deliberation layer** (generate candidates → score → select) on top of a VLA that predicts a driving trajectory (3 s, 6 waypoints) from a **single monocular front camera**, and measured whether it beats **greedy decoding**.

**The answer is no, and this repo is that negative result together with the controlled experiments that closed it.**

**All numbers are nuScenes val · deterministic crop · 600 records / 150 scenes · scene-clustered bootstrap** (`ci_sc`) and **scene-mean Wilcoxon** (`p_sc`). The metric is `avgL2@3s` (lower is better), in metres.

---

## 1. Headline

| | `avgL2@3s` |
|---|---|
| **greedy** (no deliberation) | **3.5557** |
| **GoT** (deliberation) | **3.6072** |
| difference | **+0.0397 ± 0.0105** · p_sc **0.0003** · ci_sc [+0.022, +0.083] · **positive on 3/3 seeds** |
| oracle `minADE_C` (best candidate in pool) | **2.9770 ± 0.0264** |
| mean-trajectory trivial baseline | 5.4369 |

★ **There is 16.3 % of headroom in the candidate pool, and deliberation not only fails to collect it — it does worse than greedy.** The model does use the image (3.5557 vs mean-trajectory 5.4369, −34.6 %).

## 2. Why it cannot choose — every selection rule we could build

`per_sample.csv` stores each candidate's **true L2** alongside its component scores in the same order, so **offline counterfactual evaluation of any selection rule is exact, with no approximation**. 1,800 record-plans / scene-wise 5-fold CV.

| rule | `avgL2@3s` | recovery | within-pool rho |
|---|---|---|---|
| random | 4.6560 | 0 % | — |
| `path_score` (current) | 3.5954 | 63.2 % | +0.5029 |
| kinematic alone | 3.5927 | 63.3 % | +0.5186 |
| **ridge + interactions (learned)** | **3.5845** | **63.8 %** | +0.5239 |
| gradient boosting (learned) | 3.6024 | 62.8 % | +0.5260 |
| 44-dim trajectory geometry, GBR | 3.6283 | — | **+0.5268** ← highest rho |
| **greedy (no deliberation)** | **3.5557** | **65.5 %** | — |
| oracle (`minADE_C`) | 2.9770 | 100 % | — |

★★ **Every selection rule we can build is trapped in a 1 pp band of 62.8–63.8 %, and all of them are below greedy.**

★★ **The higher the rho, the worse the selection.** The 44-dim geometry gives the project's best rho (+0.5268) while being +0.034 worse in L2. More expressive power improves the ranking and degrades the argmax — meaning **there is not even a signal to overfit**, and this was independently reproduced three times. → **The failure is in the features, not in the combination rule.**

**Signal redundancy** — the sum is worse than the best single component: kinematic +0.5225 / likelihood +0.5211 / **the two combined, `path_score`, +0.5079**.

**Likelihood really does carry independent information** (partial correlation +0.293, CI excludes 0). Yet on the half of the records where the two signals name different winners, the win rate is **exactly 0.502** (CI [0.449, 0.556]) — **it cannot discriminate at the top of the pool.**

## 3. Anatomy of the loss — variance, not bias

| | |
|---|---|
| GoT worse / better / identical vs greedy | **42.7 % / 38.7 % / 18.7 %** (nearly symmetric) |
| share of the positive loss in the worst 5 % / 20 % | **46.2 % / 84.9 %** |
| per-record `min(GoT, greedy)` ceiling | 3.4719 (greedy −0.0837 = 15.2 % of the headroom) |

**The deliberation layer does not add bias, it adds variance. In a planner, variance is pure loss with no upside.**

★★ **And it cannot even detect that it is choosing badly.** Spearman (observable proxy vs actual deficit):

| proxy | rho |
|---|---|
| `candidate_spread` | −0.0123 |
| score margin | −0.0024 |
| `worst_candidate` | −0.0231 |
| learned fallback detector (held-out) | +0.0337 |

All three are the same size as a synthetic random control. The fallback sweep is **3 proxies × 5 deferral levels × 2 held-out = all 30 cells below greedy**, and the decisive symptom is that **there is no interior optimum** — performance improves monotonically with more deferral (70 % is best), so the rule holds no information; it is simply blending back toward greedy.

## 4. Doors that are closed — interventions and their numbers

| what we tried | result | verdict |
|---|---|---|
| **drop the command component** (GT-leak check) | `d_output` −0.0002, CI [−0.029, +0.029] | no leak |
| **separate re-weighting from pruning** (`cmd_prune_only`) | `d_pool` exactly 0, output null | re-weighting cannot fix it |
| ★**Aggregate** — combine instead of choosing | `final_top3` `d_output` **−0.0206, p_sc 0.4666**, `d_pool` exactly 0 | null |
| **segment-wise fusion** (mean of m=2) | +0.2207, infeasible **41.8 %** (base rate 2.5 %) | harmful |
| ★**make the pool worse on purpose** | `d_pool` **+0.2493** yet output −0.0085 (ns) | **absorption 103 %** |
| ★**make the pool better** | absorbed identically | **absorption 103–106 %** |
| **perfect perception** (GT boxes) for collision avoidance | discrimination ceiling 23.0 %, override win rate **0.333** CI [0.136, 0.542], actual counterfactual **+0.0338 worse** | the loss is not collisions |
| ★**train-time selection** (best-of-n distillation) | displacement **−0.1238** ci_sc [−0.164, −0.082] | **the headroom is an order statistic** |
| **continuous action-head decoder** | 3.6008 vs 3.5557, +0.0452 **p_sc 0.3975** | decoder is irrelevant |
| **2 more epochs of base training** | 3.6051, +0.0494 **p_sc 0.5561** ci_sc [−0.040, +0.146] | **the base has converged** |
| **single-frame information ceiling** (probe) | constant-velocity residual `resid` skill **−0.4 %** ci_sc [−0.004, +0.021] | **SATURATED** |
| ★**output smoothing** | difference-in-differences **+0.0121** [+0.0031, +0.0233] p 0.0046 (23.4 % of the gap) | real but **not adoptable** |

★ **The absorption law** is the centre of this table — whether the pool is **improved or degraded**, the selector feeds 100 % of it back and the output does not move. This is the answer to *"can't we just make the candidates better?"*

★ **The key is that `displacement` is negative.** Good candidates are **neighbours of the trajectory the model already emits**, and the further away they are, the worse they get. So min-of-8 is *a lottery drawn around the existing output*, and distilling its winner is **what GT training already asks for**.

★ **Smoothing was only half right.** The difference is real, but **in the worst 5 % the gain is negative (−0.0160)** — the excursions that dominate the loss are not jitter. Part of the variance lives at the smoothness level; the rest is a selection problem.

## 5. The structure of the error

| 3 s error | longitudinal | lateral | ratio |
|---|---|---|---|
| greedy | **6.011 m** | 0.797 m | **7.5×** |

⚠️ **Going from here to "so the score is blind to the longitudinal axis" is wrong — that was rejected by measurement.** Of the headroom available on each axis, the fraction the score actually recovers is **57.4 % longitudinal** [52.8, 62.0] and **78.2 % lateral** [73.3, 82.7]: the score is a decent ranker on both axes. It still loses to greedy.

★★ **Even observed ego-motion cannot be recovered by selection.** Handing a perfect initial-speed oracle **to the selector alone** yields +0.0205 (ns) — because the pool itself is the problem:

| | value |
|---|---|
| first-step longitudinal pool span | **0.2703 m** |
| \|pool median − GT\| | **1.0015 m** |
| ★**coverage** (GT inside the pool's range) | **8.0 %** |

**Eight candidates cannot select an answer that lies 3.7× their own width outside them.** Apply the same information **to the trajectory** instead and you get −2.1138 (3.5557 → **1.44**) — so this is **a generator problem, not a selector problem**.

## 6. World Model (offline evaluator)

Trajectory → generate the future image → compare with the real frame. 250 records.

| | |
|---|---|
| the plan's future is less realistic than GT's | **delta +0.756**, CI [0.616, 0.896] |
| gate `wm_gain_over_copy` | 1.136 CI [0.995, 1.276] **PASS** |
| gate `sensitivity_mirror` | 0.748 CI [0.431, 1.089] **PASS** |
| ★**but it cannot rank** | **`rho(delta, L2)` = −0.0091** |

**It detects but does not order** — the same failure pattern as §2.

## 7. ★Methodological contribution — a random crop on the evaluation path

`data/item_processor.center_crop` is **a random crop despite its name** ([item_processor.py:28](data/item_processor.py#L28)), and it was **live on the evaluation path**.

| | random crop | deterministic crop |
|---|---|---|
| `codebook_l2` when loading the same frame twice | **14.15** | **0.0** |
| CI half-width | 0.0873 | **0.0306** (2.9× narrower) |
| `minADE_C` | 2.8646 | 3.0053 |

**Noise was hiding the deficit** — the earlier conclusion *"GoT ≡ greedy"* flipped to **"significantly worse"**, and we now detect 0.05 m at p = 0.0003. The crop was also **a hidden source of GoT's candidate diversity**. The old behaviour is reproducible with `--legacy_random_crop`.

★ **Lesson**: the original sanity check passed the same object twice, got 0.0, and cleared — but since the crop happens only once, that check **could not detect nondeterminism in principle**. **Always run identity checks by loading twice, independently.**

---

## Environment

```bash
conda activate VLA_GoT
```

| item | value |
|---|---|
| hardware | 3×RTX4090 (24 GB), no NVLink (PCIe) |
| ★ transformers | **pinned to `transformers==4.43.0` / `tokenizers==0.19.1`** |
| base/tokenizer | `../ckpts/Lumina-mGPT-7B-768` |
| ★ chameleon VQGAN | `../ckpts/chameleon/tokenizer/{text_tokenizer.json,vqgan.yaml,vqgan.ckpt}` |

⚠️ **The chameleon VQGAN is a required asset that is documented neither in the upstream README nor in the HF repo** (it originates from Meta). Without it, image tokenisation fails entirely. It is not included here for licensing reasons.

⚠️ If importing `model` fails with *"transformers requires torch>=2.4"*, **force-reinstall** the two pinned packages above.

## Reproduction

⚠️ `nuscenes_v1.0-trainval_{train,val}.json` are not in the repo (size). After downloading nuScenes, build them with
`python data/preprocess_nuscenes.py --dataroot <nuScenes> --version v1.0-trainval --out_dir ./data/nuscenes_records`.
**The two files you need to reproduce the numbers are included** — the eval-set definition
`nuscenes_val_scenespread.json` (which 600 records / 150 scenes) and the normalisation constants `nuscenes_norm.json`.

```bash
# 0) build the eval set (once, CPU)
#    ⚠️ Do NOT truncate with --limit N. Records are in scene order, so the first N are not a sample.
python scripts/make_turn_subset.py \
  --records ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --out ./data/nuscenes_records/nuscenes_val_scenespread.json \
  --keep left right straight --per_scene 4        # 600 records / 150 scenes

# 1) eval (3-GPU parallel runner, ~2 h)
bash scripts/run_headline.sh

# 2) deep analysis of a single run (GPU 0). The OVERALL row is the headline
python analysis/analyze_got_csv.py results/headline/ref/per_sample.csv \
  --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json

# 3) ★always compare arms paired (the CI narrows from 0.27 to 0.08)
python analysis/compare_arms.py --ref results/headline/ref/per_sample.csv \
  results/headline/ref/per_sample.csv results/headline/<arm>/per_sample.csv
```

★ **Report only `p_sc` and `ci_sc`.** `p_rec` treats records as independent and is inflated; it is printed only to show the difference between the two.

### Training (for reference — base and WM are both done)

```bash
torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path ../ckpts/Lumina-mGPT-7B-768 --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --data_config_train ./data/nuscenes_records/nuscenes_v1.0-trainval_train.json \
  --data_config_val_ind ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --data_config_val_ood ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --norm_path ./data/nuscenes_records/nuscenes_norm.json \
  --output_dir ./output/nuscenes_trainval_full_r256 \
  --trainable full --optimizer paged_adamw8bit \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --save_iteration_interval 2000 --ckpt_max_keep 4 \
  --epochs 5 --lr 2e-5 --precision bf16 --checkpointing --ft true
```

★ `--resume_path` and `--ft true` are **both required** (missing either one crashes). When resuming without `--ft true`, the 8-bit optimizer state is loaded onto the GPU in full and you **OOM**.

## Layout

| path | role |
|---|---|
| `got_drive/` | ★**the driving GoT itself** — pipeline, scoring (kinematic/command/likelihood), fusion, deterministic crop |
| `train_nuscenes*.py` · `eval_*.py` | training / evaluation |
| `analysis/` | ★**19 controlled-experiment tools** — all GPU-free, offline analysis that reads the csv |
| `scripts/` | 3-GPU parallel runners (`run_*.sh`), eval-set builder, environment checks |
| `results/` | the raw material behind the paper's tables and the WM (`per_sample.csv`, `summary.json`) |
| `data/preprocess_*.py` | nuScenes / NAVSIM record builders |
| `got_vla_v2/` | the original GoT for LIBERO (legacy, unrelated to the driving results) |

**The analysis tools are half of this repo.** Every table in §2 above was produced without a GPU — measure the ceiling here before spending GPU time on a new idea.

## ⚠️ Things that will silently mislead you if unread

1. ★**Never take a sample by position.** Records are in scene order, so "the first N" is not a sample. This is the most expensive failure mode in this project and we walked into it several times. Sample selectors should be **random, max-based, or physically defined only**.
2. ★**Do not compare against a remembered number — put the comparison in the same run.** You get a wiring check for free (it earned its keep twice).
3. **Judge by L2, not by loss.** There is a precedent where val closs rose while L2 improved by −16.6 %.
4. **On a shared GPU, launching an eval during training kills the training, not the eval.** Two concurrent evals also SIGKILL each other as their `from_pretrained` peaks overlap.

## Related repository

- **navsim_tools** — the NAVSIM (PDMS) track. It is the other half of the contrast: *"deliberation loses on a metric that asks you to match the realised future, and wins on one that asks whether the plan is good."* It lives in a separate repo because it needs a different environment (`navsim_wvg`).

## Provenance

- **base**: [RynnVLA-002 / WorldVLA](https://github.com/alibaba-damo-academy/RynnVLA-002) (built on Lumina-mGPT-7B-768). This repo is a fork of that release code.
  ⚠️ It is **not** `unified_video_action` (UVA) — conceptually similar, but a different architecture and code lineage. Cite with care.
- **GoT**: Besta et al., *Graph of Thoughts*, AAAI 2024 — [spcl/graph-of-thoughts](https://github.com/spcl/graph-of-thoughts)
  ⚠️ What is implemented here is precisely a `Generate → Score → KeepBestN` loop, i.e. **beam search (ToT)**. The official `Aggregate` (merging several thoughts into one — the only operation that makes GoT a graph) was implemented and tested separately in §4, and it was **null**.
- **V2V-GoT** ([arXiv 2509.18053](https://arxiv.org/abs/2509.18053)): uses GoT as **task decomposition** rather than search. Our negative result is about the **search axis**, so the two do not conflict.

## License

Research code. It follows the license of RynnVLA-002. Checkpoints and datasets are not included.

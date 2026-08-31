# WorldVLA-GoT — **A deliberation layer does not win** at driving planning

We put a **Graph-of-Thought-style deliberation layer** (generate candidates → score → select) on top of a VLA that predicts a driving trajectory (3 s, 6 waypoints) from a **single monocular front camera**, and measured whether it beats **greedy decoding**.

**The answer is no, and this repo is that negative result together with the controlled experiments that closed it.**

**All numbers are nuScenes val · deterministic crop · 600 records / 150 scenes · scene-clustered bootstrap** (`ci_sc`) and **scene-mean Wilcoxon** (`p_sc`). The metric is `avgL2@3s` (lower is better), in metres.

---

## 1. Headline

| | `avgL2@3s` |
|---|---|
| **greedy** (no deliberation) | **3.5557** (deterministic) |
| **GoT** (deliberation) | **3.5954 ± 0.0105** (3-seed mean; per seed 3.6072 / 3.5874 / 3.5915) |
| difference | **+0.0397 ± 0.0105** · p_sc **0.0003** · ci_sc [+0.022, +0.083] · **positive on 3/3 seeds** |
| oracle `minADE_C` (best candidate in pool) | **2.9770 ± 0.0264** |
| mean-trajectory trivial baseline | 5.4369 |

★ **There is 16.3 % of headroom in the candidate pool, and deliberation not only fails to collect it — it does worse than greedy.** The model does use the image (3.5557 vs mean-trajectory 5.4369, −34.6 %).

### 1a. Did fine-tuning teach the model anything? — the pre-training backbone, measured

Before asking whether deliberation helps, establish that there is a driving policy to deliberate over.
All rows: 600 records / 150 scenes, deterministic centre crop, seed 42.

| arm | valid output | ADE | FDE | L2 Avg. | Coll Avg. | forward | s/record | rel. cost |
|---|---|---|---|---|---|---|---|---|
| raw backbone · free generation | **0 / 600** | n/a | n/a | n/a | n/a | 1 | 0.67 † | — |
| raw backbone · grammar-forced | 600 / 600 | **21.3287** | 25.8088 | 22.1479 | 4.834 | 1 | 1.17 † | — |
| *(control)* fine-tuned · grammar-forced | 600 / 600 | **3.5557** | 6.2439 | 4.0850 | 2.445 | 1 | 1.36 † | — |
| fine-tuned · greedy | 600 / 600 | **3.5557** | 6.2439 | 4.0850 | 2.445 | 1 | 0.97 | **1×** |
| fine-tuned · GoT | 600 / 600 | 3.5954 ± 0.0105 | 6.3101 ± 0.0425 | 4.1251 ± 0.0140 | 2.371 ± 0.032 | 20 | 14.65 ± 0.26 | **15×** |
| *(reference)* stop prediction | — | 9.2230 | 15.8187 | 10.5425 | — | 0 | 0 | — |
| *(reference)* mean-trajectory prior | — | 5.4369 | 9.3211 | 6.2140 | 7.222 | 0 | 0 | — |

† grammar-forced arms decode under a different rule, so their wall-clock is not comparable with the
free-run rows; `rel. cost` is only defined against the fine-tuned greedy baseline. `s/record` also drifts
run to run with GPU contention — the *same* checkpoint and the *same* 3.5557 appear at 0.91, 0.97 and 1.36
across three runs. **Only the 15× GoT figure is a within-run comparison**, and it is the only cost claim
this table supports.

★ **The backbone emits no trajectory at all.** `<reserved10000>`-family action tokens are *empty slots* in
the Chameleon vocabulary — they never occur in pre-training — so the parser finds zero action groups.
This measures the **output protocol**, not driving ability, which is why the second row exists.

★★ **Forced to emit well-formed waypoints, the backbone is worse than predicting a full stop** — ADE 21.33
against 9.22, a factor of 2.3. Δ vs fine-tuned **+17.7731**, ci_sc [+17.16, +18.35], p_sc < 0.0001, and the
backbone wins on **0 / 600** records. There is no driving prior in the pre-trained weights to recover.

★ **The control row licenses that reading**: applying the same grammar mask to the *fine-tuned* model
reproduces ADE 3.5557 and its per-step curve exactly, so **masking costs nothing in accuracy** and 21.33 is
the backbone's own number, not an artefact of the mask. Read the two grammar-forced rows against each other
— that pair is the clean comparison.
⚠️ Grammar-forced decoding is a **separate arm** — never put it in the same column as a free-run number
without its control.

★ **The per-step curve separates "weak prior" from "no prior"**: the backbone goes 18.62 → 22.02 → 25.81
(×1.39) — already 18.6 m wrong at 1 s and nearly flat — while the fine-tuned model accumulates normally at
1.99 → 4.02 → 6.24 (×3.14). Picking bins at random over the normalised range has an expectation of ≈ 27.7 m,
which puts the GT roughly 18–20 m away: exactly the 18.62 observed.

→ **Read the two tables together**: fine-tuning moves ADE 21.33 → 3.5557, and deliberation on top of that
moves it 3.5557 → 3.5954 in the wrong direction, for 15× the cost.

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

★★★ **A training-free constant-velocity extrapolation beats this model** (`analysis/eval_constant_velocity.py`, 450 records / 150 scenes, **CPU only**). Read v0 as the previous adjacent keyframe's longitudinal displacement and hold it straight for 3 s. No model, no image, no learning.

| same 450 records | ADE (`avgL2@3s`) | FDE (`L2@3s`) | avg. Col. (UniAD) | avg. Col. (ST-P3) |
|---|---|---|---|---|
| VLA (finetuned) | 3.5272 | — | 2.519 % | 0.926 % |
| **constant velocity** | **1.2962** | **2.7316** | **0.519 %** | **0.111 %** |

Δ ADE **−2.2310**, ci_sc [−2.6352, −1.8508], p_sc <0.0001, constant-velocity win rate **76.2 %**. **It wins on collision as well as on L2.**

⚠️ **This is a statement about the benchmark, not a defect in the method.** It is BEV-Planner's critique of nuScenes open-loop reproduced independently inside our pipeline, and it measures **the same thing as the 1.44 above, from the other side**. This project **does not consume ego status by design**, which is exactly what makes that shortcut unavailable to it — the absolute L2 is the price.

★ **Why 450/600**: the other 150 records **open their scene**, so no predecessor exists in any source (attainable coverage 100 %). Both arms above are scored on the **same 450**, and the model's collision rate is 2.445 % on all 600 vs 2.519 % on the 450 — so that subset is not skewed.

★★ **The contrast in the other direction — the backbone holds no driving prior.** Checked in two stages.

**(1) Free generation.** Running the finetuning's own starting point (`../ckpts/Lumina-mGPT-7B-768`) on the
same 600 records gives `n_malformed_generation` **600 / 600**, `n_evaluated` **0**. The `<reserved10000>`
family are *empty vocabulary slots* that never appear in pretraining, so the parser finds no action group.
That number measures whether the model knows the OUTPUT PROTOCOL, not whether it can drive.

**(2) Grammar forced (`--constrained`).** Mask the logits to the valid tokens at every step and the backbone
also emits 600/600 well-formed trajectories. The result is **ADE 21.3287**.

| same 600 records | ADE | verdict |
|---|---|---|
| raw backbone (grammar forced) | **21.3287** | **2.3x worse** than predicting no motion |
| stay-still (all-zero waypoints) | 9.2230 | |
| mean-trajectory prior | 5.4369 | |
| VLA (finetuned) | **3.5557** | |

Delta ADE **+17.7731**, ci_sc [+17.1630, +18.3462], p_sc <0.0001, and the backbone wins **0 of 600** records.

★ **The control licenses that reading**: forcing the same grammar on the FINETUNED model returns ADE
**3.5557, identical to free generation to the last decimal** (`per_step_L2` matches too). Masking costs
nothing in accuracy, so the backbone's 21.33 is the model's own value, not an artefact of the decoding rule.

→ Read the two contrasts together: **finetuning is what makes the task possible at all (0 % -> 100 %), and the
absolute accuracy it buys still loses to a constant-velocity extrapolation that only knows the ego speed.**

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
# 4) training-free constant-velocity baseline (CPU only, seconds). Runs while the GPUs are busy
#    ⚠️ --records must be the FULL val json (consecutive keyframes). The eval set is ~5.5 s apart,
#       so it contains no predecessors of its own
python analysis/eval_constant_velocity.py \
  --records ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
  --eval_records ./data/nuscenes_records/nuscenes_val_scenespread.json \
  --output_dir ./results/const_velocity \
  --ref ./results/base_ckpt/incumbent_cont2_ep1 \
  --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json \
  --ref_collision_csv ./results/headline/ref/per_sample.csv \
  --dt_lo 0.3 --dt_hi 0.8
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
| `analysis/` | ★**20 controlled-experiment tools** — all GPU-free, offline analysis that reads the csv |
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
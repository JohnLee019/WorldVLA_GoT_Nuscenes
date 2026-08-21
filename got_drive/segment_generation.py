"""
Segment-wise waypoint generation for driving GoT (open-loop nuScenes).

The base eval (`eval_nuscenes.predict_waypoints`) generates all `time_horizon`
waypoints in a single free run. Graph-of-Thought planning instead needs to:

  (1) condition generation on a PREFIX of already-decided waypoints, and
  (2) generate only the NEXT segment, so several candidate continuations can be
      scored and the best committed before moving on.

Both reduce to one primitive: feed the model the standard image+prompt context
followed by k already-tokenized waypoint groups, and let it continue.

Why this works without any model change
---------------------------------------
Training sequences are
    <bos> prompt <|image|>... <sep>  [wp_1][wp_2]...[wp_H]  <sep>
where each [wp] group is `[10004, bin_x, bin_y, 15004]` and <sep> is 8710
(`<reserved08706>`, the assistant-turn terminator). The inference context built
by `eval_nuscenes.predict_waypoints` ends exactly at the first <sep>, i.e. right
where the model is trained to start emitting [wp_1]. Appending the first k
ground-decided groups in that same format therefore lands us precisely where the
model expects [wp_{k+1}] next. `generate_dis_ma` slices off everything up to
`input_ids.shape[1]`, so the returned groups are the continuation only -- the
prefix is never echoed back.

This is the (A) building block. If the model cannot coherently continue a
partial trajectory here, segment-wise GoT is not viable on the current
checkpoint and training must instead be made segment-aware.
"""

import numpy as np
import torch
from transformers import GenerationConfig

# Assistant-turn terminator (`<reserved08706>`); reused as the generation EOS,
# matching eval_nuscenes.predict_waypoints.
SEP_EOS_TOKEN = 8710
ACTION_START_TOKEN = 10004   # <reserved10000>
ACTION_END_TOKEN = 15004     # <reserved15000>


def unnorm_waypoints(norm_wp, wp_min, wp_max):
    """[-1, 1] bin space -> metres. Mirror of eval_nuscenes.unnorm_waypoints."""
    return (norm_wp + 1) / 2 * (wp_max - wp_min + 1e-8) + wp_min


def tokenize_waypoint_prefix(item_processor, prefix_wp):
    """prefix_wp: (k, 2) metres -> flat list of action-group token ids.

    Tokenizes each waypoint with `item_processor.process_action`, so the layout
    and normalisation are byte-for-byte identical to what training produced.
    """
    ids = []
    for wp in np.asarray(prefix_wp, dtype=np.float64):
        ids += item_processor.process_action(wp)["input_ids"]
    return ids


def build_context_ids(model, item_processor, image, prompt, prefix_wp=None):
    """Image+prompt context, optionally extended with k prefix waypoint groups.

    With prefix_wp=None this is exactly the context from
    `eval_nuscenes.predict_waypoints`.
    """
    conv = {
        "conversations": [{"from": "human", "value": prompt + "<|image|>"}],
        "image": [image],
        "action": [],
    }
    tokens = item_processor.process_item(conv, training_mode=False)
    if prefix_wp is not None and len(prefix_wp) > 0:
        tokens = tokens + tokenize_waypoint_prefix(item_processor, prefix_wp)
    return torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)


@torch.no_grad()
def predict_segment(model, item_processor, image, prompt, args,
                    prefix_wp=None, n_generate=None,
                    temperature=1.0, do_sample=False):
    """Generate the next `n_generate` waypoints given `prefix_wp` already decided.

    Returns (n_generate, 2) metres, or None if the continuation was malformed
    (fewer well-formed action groups than requested, or wrong action_dim).

    With prefix_wp=None and n_generate=time_horizon this reduces to the free-run
    baseline of eval_nuscenes.predict_waypoints (greedy when do_sample=False).
    """
    n_prefix = 0 if prefix_wp is None else len(prefix_wp)
    if n_generate is None:
        n_generate = args.time_horizon - n_prefix
    if n_generate <= 0:
        return np.empty((0, args.action_dim), dtype=np.float64)

    input_ids = build_context_ids(model, item_processor, image, prompt, prefix_wp)

    # each waypoint emits [start, x, y, end]; +8 leaves headroom for a stray token
    max_new = n_generate * (args.action_dim + 2) + 8
    generation_config = GenerationConfig(
        max_new_tokens=max_new,
        max_length=model.config.max_position_embeddings,
        temperature=temperature,
        top_k=None,
        do_sample=do_sample,
        eos_token_id=[SEP_EOS_TOKEN],
    )

    # generate_dis_ma parses the stream into [10004 ... 15004] groups. An
    # undertrained model can emit a malformed stream (e.g. an end marker before
    # any start), which raises inside the parser; treat that as a malformed
    # generation rather than killing the run, exactly like the base eval.
    try:
        groups = model.generate_dis_ma(input_ids, generation_config)
    except Exception:
        return None

    if len(groups) < n_generate:
        return None
    groups = groups[:n_generate]  # ignore any trailing over-generation
    norm_wp = []
    for g in groups:
        g = g.detach().float().cpu().numpy()
        if g.shape[0] != args.action_dim:
            return None
        norm_wp.append(g)
    norm_wp = np.stack(norm_wp, axis=0)
    return unnorm_waypoints(norm_wp, item_processor.wp_min, item_processor.wp_max)


# --------------------------------------------------------------------------- #
# Model self-likelihood: the only scene-grounded candidate score available
# without training anything else.
#
# Every score tried so far (scoring_driving.score_kinematic / score_command) is
# computed from the waypoints alone and never looks at the image, so none of them
# can tell whether a smooth, command-consistent trajectory suits THIS scene. The
# model's own conditional log-likelihood of a candidate, given the same
# image+prompt context it was trained on, is grounded in the pixels and costs one
# forward pass -- no second network, no world model, no extra checkpoint.
#
# Caveat to keep in mind when reading the result: the candidates were sampled
# from this very model at temperature > 1, so ranking them by its likelihood
# partly undoes the sampling and may collapse towards greedy. That outcome is
# still informative -- it would say the generator cannot identify which of its
# own samples are good -- but it is the expected null, not a surprise.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def trajectory_logprob(model, item_processor, image, prompt, traj, ctx=None):
    """Mean per-token log-likelihood of `traj` under the model, given image+prompt.

    Higher = the model finds this trajectory more likely. Returns None if the
    trajectory tokenizes to nothing.

    Implemented by reusing the model's OWN training-path loss rather than
    gathering logits by hand: with labels masked to -100 everywhere except the
    action tokens, `c_loss` is exactly the mean cross-entropy over those tokens,
    so the shift and masking conventions match training by construction instead
    of by a hand-written off-by-one. The eval-path forward is unusable here --
    it accumulates `init_input_ids` across calls and slices the attention mask
    to its last row, both of which assume incremental decoding.

    All candidates for one record tokenize to the same length (time_horizon
    groups of action_dim+2 tokens), so mean vs sum does not change the ranking;
    mean is returned because it is comparable across records.

    `ctx` is the prebuilt image+prompt context from `build_context_ids`. Pass it
    when scoring several candidates for the SAME record: the context is identical
    across them, and building it re-runs the VQGAN image tokenizer every time --
    |C| redundant image encodings per record, which dwarfs the cheap part (one
    forward per candidate). Left as None it is built here, which keeps the
    function usable standalone.
    """
    if ctx is None:
        ctx = build_context_ids(model, item_processor, image, prompt)   # (1, L)
    act = tokenize_waypoint_prefix(item_processor, traj)
    if not act:
        return None
    ids = ctx[0].tolist() + act
    labels = [-100] * ctx.shape[1] + act
    c_loss, _ = model(input_ids=[ids], labels=[labels], training=True)
    return -float(c_loss)


def make_likelihood_score_fn(model, item_processor, prompt, args):
    """lik_score_fn(image, traj) -> float, for DriveGoTPipeline's final re-rank.

    Signature matches the pipeline's injected-scorer convention (see
    wm_score_fn); failures surface as None and the ranker treats them as
    neutral rather than poisoning the candidate set.

    Caches the image+prompt context per image object, because the pipeline calls
    this once per candidate with the SAME image and building the context re-runs
    the VQGAN image tokenizer -- |C| redundant encodings per record. Keyed by
    id(image) with a single-entry cache: the pipeline holds one PIL image for the
    whole plan and moves to the next record afterwards, so a one-slot cache hits
    every time within a record and never grows.
    """
    cache = {"key": None, "ctx": None}

    def lik_score_fn(image, traj):
        key = id(image)
        if cache["key"] != key:
            cache["key"] = key
            cache["ctx"] = build_context_ids(model, item_processor, image, prompt)
        return trajectory_logprob(model, item_processor, image, prompt, traj,
                                  ctx=cache["ctx"])

    return lik_score_fn

"""
World-model inference for driving GoT: predict the next front-camera frame.

Wraps the WM trained by train_nuscenes_wm.py -- `(current frame + action)
-> future frame` -- into a single call the GoT pipeline can use for Context
Update. It mirrors got_vla_v2/scoring/score_world_model.py (the LIBERO WM
readout) but:

  * uses the driving WM conversation format (single front image + k waypoints),
    matching data/dataset_nuscenes_wm.py exactly, and
  * returns the decoded PIL frame itself (Context Update needs the image), not a
    pixel-difference score.

The WM must have been loaded with mask_image_logits=False; otherwise generate_img
produces no valid image tokens.
"""
import numpy as np
import torch
from transformers import GenerationConfig

from data.dataset_nuscenes_wm import WM_PROMPT

IMG_START_TOKEN = 8197
IMG_END_TOKEN = 8196
SEP_EOS_TOKEN = 8710


@torch.no_grad()
def predict_next_frame(world_model, item_processor, current_img, action_wp, device,
                       max_new_tokens=700, do_sample=False, temperature=1.0,
                       top_k=None):
    """(current_img, action waypoints) -> predicted next PIL frame, or None.

    action_wp: (k, 2) metres in the CURRENT ego frame -- the segment just
    committed. Format is identical to a WM training example's human turn.

    do_sample=False (greedy) is the DEFAULT and matches RynnVLA-002/WorldVLA
    upstream, which ships the sampling branch commented out (handoff sec.1.7c).
    Every number in sec.1 was produced with greedy, so changing this default
    would silently invalidate them.

    E4 exists because the object duplication in generated frames looks like a
    greedy-AR repetition artefact rather than a limit of the 2-epoch model --
    sec.1.7c's "there is no knob" was too strong, this is the knob. Sampling
    can also inject high-frequency detail that makes codebook_l2 WORSE, so if
    it is ever adopted the delta and all three gates have to be re-measured
    (~6 GPU-h). Until then it is a figure-only switch.
    """
    k = len(action_wp)
    conv = {
        "conversations": [{
            "from": "human",
            "value": WM_PROMPT + "<|image|>" + "<|action|>" * k,
        }],
        "image": [current_img],
        "action": [list(map(float, wp)) for wp in np.asarray(action_wp, dtype=np.float64)],
    }

    saved_init = getattr(world_model, "init_input_ids", None)
    world_model.init_input_ids = None
    try:
        tokens = item_processor.process_item(conv, training_mode=False)
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=world_model.config.max_position_embeddings,
            temperature=temperature, top_k=top_k, do_sample=do_sample,
            eos_token_id=[SEP_EOS_TOKEN],
        )
        gen = world_model.generate_img(input_ids, gen_cfg)[0]

        starts = torch.where(gen == IMG_START_TOKEN)[0]
        ends = torch.where(gen == IMG_END_TOKEN)[0]
        if len(starts) == 0 or len(ends) == 0:
            return None
        frame_tokens = gen[starts[0]:ends[0] + 1].cpu().tolist()
        try:
            return item_processor.decode_image(frame_tokens)
        except Exception:
            return None
    finally:
        world_model.init_input_ids = saved_init


def make_wm_context_update_fn(world_model, item_processor, device):
    """Bind the WM into context_update_fn(current_img, segment_wp) -> next_img.

    Returns None on WM failure; the pipeline falls back to reusing the current
    frame so a bad WM prediction degrades to 'no context update' rather than
    crashing the plan.
    """
    def context_update_fn(current_img, segment_wp):
        nxt = predict_next_frame(world_model, item_processor, current_img, segment_wp, device)
        return nxt if nxt is not None else current_img

    return context_update_fn


# ──────────────────────────────────────────────────────────────────────────
# Driving WM SCORE readout  (the research contribution)
# ──────────────────────────────────────────────────────────────────────────
#
# The LIBERO WM score (got_vla_v2/scoring/score_world_model.py) rewards LARGE
# pixel change between the predicted future frame and the current one -- a
# manipulation prior ("make the gripper move / change the scene") that is exactly
# backwards for driving, where reckless motion is bad.
#
# Reframe for driving:
#     good action  ->  a future the world model can predict CONFIDENTLY
#                      (an in-distribution, plausible scene evolution)
#     bad  action  ->  drives the scene out of distribution, so the WM's
#                      per-token predictive distribution over the future frame
#                      becomes UNCERTAIN.
#
# So we score the NEGATIVE MEAN PREDICTIVE ENTROPY of the WM over the generated
# future-frame tokens: lower entropy (more confident future) => more plausible
# action => higher score.
#
# Why entropy, not the NLL of the greedily-chosen tokens: greedy NLL is biased
# low for EVERY candidate (greedy always picks a high-prob token), so it barely
# discriminates. Entropy is a property of the whole distribution at each step, so
# it separates "the WM knows what comes next" from "the WM is guessing" even
# under greedy decoding. (The chosen-token NLL is returned too, for ablation.)

@torch.no_grad()
def score_next_frame_plausibility(world_model, item_processor, current_img, action_wp, device,
                                  max_new_tokens=700, return_components=False):
    """(current_img, candidate action) -> WM plausibility scalar (higher = better).

    Returns None if the WM did not emit a well-formed frame (treated as neutral
    by the ranker). action_wp: (k, 2) metres in the CURRENT ego frame -- so this
    is cleanest in Mode B, where each segment is expressed in its own frame that
    matches `current_img`; in Mode A (fixed t0 image) it is an approximation.

    If return_components, returns dict(plausibility, entropy, nll); else the
    scalar plausibility (= -mean_entropy).
    """
    from model import ChameleonForConditionalGeneration  # local import: avoid top-level cycle

    k = len(action_wp)
    conv = {
        "conversations": [{
            "from": "human",
            "value": WM_PROMPT + "<|image|>" + "<|action|>" * k,
        }],
        "image": [current_img],
        "action": [list(map(float, wp)) for wp in np.asarray(action_wp, dtype=np.float64)],
    }

    saved_init = getattr(world_model, "init_input_ids", None)
    world_model.init_input_ids = None
    world_model._force_no_att_mask = True
    try:
        tokens = item_processor.process_item(conv, training_mode=False)
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=world_model.config.max_position_embeddings,
            temperature=1.0, top_k=None, do_sample=False,
            eos_token_id=[SEP_EOS_TOKEN],
        )
        # Same call generate_img makes, plus output_scores so we get the per-step
        # predictive distribution. Model class is untouched.
        res = ChameleonForConditionalGeneration.generate(
            world_model, input_ids=input_ids, generation_config=gen_cfg,
            output_hidden_states=True, return_dict_in_generate=True,
            output_scores=True, att_mask=None,
        )
    finally:
        world_model._force_no_att_mask = False
        world_model.init_input_ids = saved_init

    gen = res.sequences[0, input_ids.shape[1]:]
    scores = res.scores  # tuple(len = n_generated) of (1, vocab); scores[t] made gen[t]
    if scores is None or len(scores) == 0:
        return None

    # image-content span [first IMG_START .. first IMG_END] within the generation
    starts = (gen == IMG_START_TOKEN).nonzero(as_tuple=True)[0]
    ends = (gen == IMG_END_TOKEN).nonzero(as_tuple=True)[0]
    if len(starts) == 0 or len(ends) == 0:
        return None
    s, e = int(starts[0]), int(ends[0])
    if e <= s:
        return None

    ent_list, nll_list = [], []
    for t in range(s, min(e + 1, len(scores))):
        logits = scores[t][0].float()
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent_list.append(float(-(p * logp).sum()))
        nll_list.append(float(-logp[int(gen[t])]))
    if not ent_list:
        return None

    mean_entropy = float(np.mean(ent_list))
    plausibility = -mean_entropy   # confident future => higher score
    if return_components:
        return {"plausibility": plausibility,
                "entropy": mean_entropy,
                "nll": float(np.mean(nll_list))}
    return plausibility


def make_wm_score_fn(world_model, item_processor, device):
    """Bind the WM into wm_score_fn(image, segment_wp) -> plausibility float|None.

    Plugs into DriveGoTPipeline(wm_score_fn=...): the pipeline calls it on the
    candidates it cheaply short-listed each segment (two-stage rerank).
    """
    def wm_score_fn(image, segment_wp):
        return score_next_frame_plausibility(world_model, item_processor, image, segment_wp, device)

    return wm_score_fn

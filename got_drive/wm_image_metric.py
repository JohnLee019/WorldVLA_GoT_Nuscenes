"""
Image-space distance between a world-model-predicted frame and the real frame.

This is the metric core of the offline WM-image evaluation (PROJECT_HANDOFF §7.4 /
§2③④). The world model renders the future CAM_FRONT frame the ego *would* see
after executing a candidate action; we then need a number that says how close
that predicted future is to what nuScenes actually recorded.

Why VQGAN CODEBOOK-VECTOR distance, not raw-pixel or token-ID match
-------------------------------------------------------------------
* Raw pixel MAE is dominated by the WM's lossy texture/background reconstruction,
  which is (approximately) trajectory-independent -- a common floor that swamps
  the viewpoint/ego-pose signal we care about (see the §2 "background cancels"
  argument: we always read this RELATIVE to the GT-action prediction, so the
  common floor drops out). Pixel MAE is kept only as a sanity baseline.
* Raw token-ID agreement ("what fraction of VQGAN tokens match") is brittle:
  two perceptually-near tokens count as a total miss, so the metric is almost
  binary and high-variance.
* The VQGAN codebook is a learned *perceptual* embedding: nearby codebook
  vectors decode to similar patches. Measuring distance between the codebook
  VECTORS of the predicted vs real tokens is a smooth, perceptually-graded
  signal -- the intended metric.

What this module exposes
------------------------
    vqgan_token_embedding_distance(item_processor, pred_img, real_img, ...)
        mean over spatial positions of the distance between the codebook vector
        the real frame quantises to and the one the predicted frame quantises to.
    pixel_mae(pred_img, real_img)                      # sanity baseline
    frame_distances(item_processor, pred_img, real_img)  # both, as a dict

The images are re-encoded through the SAME VQGAN the item_processor already owns
(item_processor.chameleon_ori_image_tokenizer), so no extra checkpoint is loaded.
The real frame is resized to the prediction's exact size first, so both produce
the identical latent grid and the per-position comparison is well defined.

NOTE (not yet run end-to-end): a trained world model does not exist yet (the old
one was discarded, retrain pending), so this has only been unit-tested on the
aggregation math (see `_selftest`). Run `python -m got_drive.wm_image_metric` on
a box with torch to exercise the self-test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────
# codebook-vector distance
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _codebook_vectors(image_tokenizer, pil_img: Image.Image) -> torch.Tensor:
    """PIL frame -> (N, e_dim) tensor of the codebook vectors it quantises to.

    Uses the VQGAN's OWN raw quantiser indices (min_encoding_indices from
    img_tokens_from_pil), which index straight into quantize.embedding -- no
    BPE / img2bp2 translation is involved, so this is the true codebook space.
    """
    idx = image_tokenizer.img_tokens_from_pil(pil_img)          # (N,) long, on vq device
    idx = torch.as_tensor(idx).flatten().long()
    emb = image_tokenizer._vq_model.quantize.embedding           # nn.Embedding(n_e, e_dim)
    idx = idx.to(emb.weight.device)
    return emb(idx)                                              # (N, e_dim)


@torch.no_grad()
def vqgan_token_embedding_distance(
    item_processor,
    pred_img: Optional[Image.Image],
    real_img: Image.Image,
    metric: str = "l2",
) -> Optional[float]:
    """Mean codebook-vector distance between `pred_img` and `real_img`.

    metric:
        "l2"     -> mean over positions of the Euclidean distance between the two
                    frames' codebook vectors (0 iff every position quantises to
                    the same code; scale is the codebook's own).
        "cosine" -> mean over positions of (1 - cosine_similarity), in [0, 2];
                    scale-free, robust to the codebook's absolute magnitude.

    Returns None if `pred_img` is None (the WM emitted no well-formed frame), so
    the caller can drop the sample rather than count a spurious 0/large value.
    The real frame is resized to the prediction's size so both quantise to the
    same-shaped latent grid.
    """
    if pred_img is None:
        return None
    tok = item_processor.chameleon_ori_image_tokenizer

    # align grids: identical H,W -> identical latent layout -> positions comparable
    real_aligned = real_img if real_img.size == pred_img.size else real_img.resize(
        pred_img.size, Image.BICUBIC)

    vp = _codebook_vectors(tok, pred_img).float()
    vr = _codebook_vectors(tok, real_aligned).float()
    if vp.shape != vr.shape:
        # var_center_crop is deterministic for a fixed size, so this should not
        # happen; guard rather than silently compare misaligned grids.
        return None

    if metric == "cosine":
        cos = torch.nn.functional.cosine_similarity(vp, vr, dim=-1)  # (N,)
        return float((1.0 - cos).mean())
    if metric == "l2":
        return float(torch.linalg.vector_norm(vp - vr, dim=-1).mean())
    raise ValueError(f"unknown metric {metric!r} (use 'l2' or 'cosine')")


# ──────────────────────────────────────────────────────────────────────────
# pixel-MAE sanity baseline
# ──────────────────────────────────────────────────────────────────────────

def pixel_mae(pred_img: Optional[Image.Image], real_img: Image.Image) -> Optional[float]:
    """Mean absolute pixel difference in [0, 1] (real resized to pred size).

    Kept only as a baseline / sanity check against the codebook metric; it is
    dominated by the WM's texture reconstruction error (see module docstring).
    """
    if pred_img is None:
        return None
    real_aligned = real_img if real_img.size == pred_img.size else real_img.resize(
        pred_img.size, Image.BICUBIC)
    a = np.asarray(pred_img.convert("RGB"), dtype=np.float32)
    b = np.asarray(real_aligned.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def frame_distances(item_processor, pred_img, real_img) -> dict:
    """All frame distances at once: {codebook_l2, codebook_cosine, pixel_mae}.

    Values are None when pred_img is None. Convenience wrapper for the eval loop
    so predicted/GT frames go through exactly one code path.
    """
    return {
        "codebook_l2": vqgan_token_embedding_distance(item_processor, pred_img, real_img, "l2"),
        "codebook_cosine": vqgan_token_embedding_distance(item_processor, pred_img, real_img, "cosine"),
        "pixel_mae": pixel_mae(pred_img, real_img),
    }


# ──────────────────────────────────────────────────────────────────────────
# self-test: aggregation math only, no VQGAN / GPU
# ──────────────────────────────────────────────────────────────────────────

class _FakeQuantize:
    def __init__(self, emb):
        self.embedding = emb


class _FakeVQ:
    def __init__(self, emb):
        self.quantize = _FakeQuantize(emb)


class _FakeTokenizer:
    """Stand-in for chameleon_ori_image_tokenizer: maps a PIL frame to a fixed
    index sequence keyed by the frame's tag, over a tiny hand-built codebook."""
    def __init__(self):
        # 4-entry codebook, 3-dim: orthogonal-ish so distances are checkable
        w = torch.tensor([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [3.0, 4.0, 0.0]], dtype=torch.float32)
        emb = torch.nn.Embedding.from_pretrained(w, freeze=True)
        self._vq_model = _FakeVQ(emb)
        self._table = {}   # frame tag -> index list

    def register(self, tag, idxs):
        self._table[tag] = list(idxs)

    def img_tokens_from_pil(self, pil_img):
        return torch.tensor(self._table[pil_img.info["tag"]], dtype=torch.long)


class _FakeProcessor:
    def __init__(self, tok):
        self.chameleon_ori_image_tokenizer = tok


def _tagged(tag, size=(8, 8)):
    im = Image.new("RGB", size)
    im.info["tag"] = tag
    return im


def _selftest():
    tok = _FakeTokenizer()
    ip = _FakeProcessor(tok)

    # identical token streams -> zero distance
    tok.register("a", [1, 2, 1, 2])
    tok.register("b", [1, 2, 1, 2])
    A, B = _tagged("a"), _tagged("b")
    d = vqgan_token_embedding_distance(ip, A, B, "l2")
    assert abs(d) < 1e-6, f"identical frames must be 0, got {d}"
    c = vqgan_token_embedding_distance(ip, A, B, "cosine")
    assert abs(c) < 1e-6, f"identical cosine must be 0, got {c}"

    # one position differs: code1=(1,0,0) vs code2=(0,1,0) -> l2 = sqrt(2) at that
    # position, 0 elsewhere; mean over 4 positions = sqrt(2)/4
    tok.register("d", [1, 2, 2, 2])   # differs from "b" only at position 0
    D = _tagged("d")
    d = vqgan_token_embedding_distance(ip, D, B, "l2")
    exp = (2 ** 0.5) / 4
    assert abs(d - exp) < 1e-6, f"expected {exp}, got {d}"

    # cosine of orthogonal unit codes (1,0,0) vs (0,1,0) is 0 -> (1-0)=1 at that
    # position, 0 elsewhere; mean = 1/4
    c = vqgan_token_embedding_distance(ip, D, B, "cosine")
    assert abs(c - 0.25) < 1e-6, f"expected 0.25, got {c}"

    # None prediction propagates as None (dropped by caller)
    assert vqgan_token_embedding_distance(ip, None, B, "l2") is None
    assert pixel_mae(None, B) is None

    # pixel_mae of two solid frames of known colours
    p = Image.new("RGB", (4, 4), (255, 255, 255))
    q = Image.new("RGB", (4, 4), (0, 0, 0))
    assert abs(pixel_mae(p, q) - 1.0) < 1e-6

    print("wm_image_metric self-test: OK")


if __name__ == "__main__":
    _selftest()

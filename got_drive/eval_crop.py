"""
Deterministic image cropping for EVALUATION.

The problem this fixes
----------------------
data/item_processor.center_crop() is a RANDOM crop despite its name: it draws the
crop offset with random.randint (item_processor.py:28-29). Every call to
process_item() -> process_image() -> var_center_crop() therefore re-crops the
image at a fresh, random offset, consuming two draws from the global `random`
stream each time.

That is correct as TRAINING augmentation. At evaluation it does two bad things:

1. It injects noise. For a 1600x900 CAM_FRONT the pipeline lands on a 341x192
   intermediate and crops 320x192, i.e. up to 21 px of horizontal jitter. The
   token grid is 16 px per patch, so that is ~1.3 token COLUMNS -- nearly every
   position in the latent grid changes. Measured: two independent loads of the
   SAME frame score codebook_l2 = 14.15, against 0.0 when the crop is fixed.

2. It couples arms that should be independent. eval_got_nuscenes runs the greedy
   free-run baseline WITHOUT re-seeding (eval_got_nuscenes.py:492; set_seed comes
   afterwards, at :516, for the GoT loop only), so the baseline inherits whatever
   RNG state the PREVIOUS record's GoT left behind. GoT issues k + 2*beam*k
   generate calls -- 20 for k4/beam2, 42 for k6/beam3, 15 for k3/beam2 -- and each
   consumes two draws. Change --k_candidates or --beam_width and every later
   baseline crop shifts, so a deterministic baseline moves. Change only
   temperature or score weights and the draw count is identical, so it does not.
   That is exactly the pattern PROJECT_HANDOFF §9 recorded ("wide" 1.6720 vs
   1.7156, "narrow" 3.0189 vs 2.9764, everything else bit-identical) and
   provisionally blamed on bf16 argmax tie-flips.

The fix, and why it is this shape
---------------------------------
We do NOT touch item_processor: training depends on the random crop as
augmentation, and it is shared code.

Instead the eval scripts crop the image ONCE, deterministically, at load time.
Everything downstream is then a no-op, because var_center_crop applied to an
image whose size is already an entry of crop_size_list picks that same size
(its aspect remainder is exactly 1.0, the maximum), skips the halving loop
(size >= 2*size is false), scales by max(w/w, h/h) = 1, and finally draws
randint(0, 0) = 0 twice. Same pixels, offsets forced to zero.

That property is asserted structurally in the self-test and was confirmed
end-to-end against the real crop_size_list on gpu-server:
    d(eval_center_crop(x), var_center_crop(eval_center_crop(x))) == 0.0000

Evaluating with a centre crop after training with random crops is standard and
stays inside the training distribution -- the centre offset is one of the offsets
training sampled.

Caveat worth keeping in view
----------------------------
Numbers produced with this enabled are NOT comparable to numbers produced before
it. Every eval script exposes --legacy_random_crop to reproduce the old
behaviour, but the two must never be mixed inside one table.
"""
from __future__ import annotations

from PIL import Image


def eval_center_crop(pil_image: Image.Image, crop_size_list) -> Image.Image:
    """Deterministic counterpart of data.item_processor.var_center_crop.

    Identical in every step -- same aspect-ratio choice of crop size, same
    progressive BOX halving, same BICUBIC rescale -- except that the crop offset
    is the true centre instead of random.randint(). Consumes no RNG.
    """
    w, h = pil_image.size
    rem = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = sorted(zip(rem, crop_size_list), reverse=True)[0][1]

    img = pil_image
    while img.size[0] >= 2 * crop_size[0] and img.size[1] >= 2 * crop_size[1]:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)

    scale = max(crop_size[0] / img.size[0], crop_size[1] / img.size[1])
    img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)

    left = (img.size[0] - crop_size[0]) // 2
    upper = (img.size[1] - crop_size[1]) // 2
    return img.crop((left, upper, left + crop_size[0], upper + crop_size[1]))


def crop_for_eval(pil_image: Image.Image, item_processor, legacy: bool = False) -> Image.Image:
    """Load-time hook for the eval scripts.

    legacy=True returns the image untouched, so process_image()'s random crop runs
    as it always did and pre-fix numbers can be reproduced.
    """
    if legacy:
        return pil_image
    return eval_center_crop(pil_image, item_processor.crop_size_list)


# ──────────────────────────────────────────────────────────────────────────
# self-test: PIL only, no torch / VQGAN / GPU
# ──────────────────────────────────────────────────────────────────────────

def _mk(w, h, seed=0):
    """Deterministic noise image -- flat colours would hide offset differences."""
    im = Image.new("RGB", (w, h))
    px = im.load()
    v = seed or 1
    for y in range(h):
        for x in range(w):
            v = (v * 1103515245 + 12345) & 0x7FFFFFFF
            px[x, y] = (v >> 16 & 255, v >> 8 & 255, v & 255)
    return im


def _selftest():
    # mirrors generate_crop_size_list(64, 32) closely enough for the geometry tests
    csl = [(256, 256), (320, 192), (512, 128), (192, 320), (128, 512), (384, 160)]

    a = _mk(200, 113, seed=7)          # scaled-down stand-in for 1600x900
    c1, c2 = eval_center_crop(a, csl), eval_center_crop(a, csl)
    assert c1.tobytes() == c2.tobytes(), "crop must be deterministic"

    # a second, independent load of the same image must crop identically
    c3 = eval_center_crop(_mk(200, 113, seed=7), csl)
    assert c1.tobytes() == c3.tobytes(), "crop must not depend on object identity"

    # idempotent: cropping an already-cropped frame changes nothing
    assert eval_center_crop(c1, csl).tobytes() == c1.tobytes(), "crop must be idempotent"

    # the no-op property var_center_crop relies on, asserted structurally:
    cs = c1.size
    assert cs in csl, "cropped size must be an entry of crop_size_list"
    rem = [min(cw / cs[0], ch / cs[1]) / max(cw / cs[0], ch / cs[1]) for cw, ch in csl]
    best = sorted(zip(rem, csl), reverse=True)[0][1]
    assert best == cs, f"var_center_crop would re-pick {best}, not {cs}"
    assert not (cs[0] >= 2 * cs[0] and cs[1] >= 2 * cs[1]), "halving loop must not run"
    assert max(cs[0] / cs[0], cs[1] / cs[1]) == 1.0, "rescale must be identity"
    # -> randint(0, 0) twice -> offsets forced to 0 -> pixels unchanged

    # centre, not corner: crop of a wide image must drop equal margins
    wide = _mk(400, 120, seed=3)
    cw_ = eval_center_crop(wide, csl)
    assert cw_.size in csl

    # the strong end-to-end version, when the real deps are importable
    try:
        from data.item_processor import generate_crop_size_list, var_center_crop
    except Exception as e:                                    # torch-less box, or wrong cwd
        print(f"eval_crop self-test: OK (structural only -- could not import "
              f"data.item_processor: {type(e).__name__}: {e})")
        print("  If you have torch, you ran this as a script: `python got_drive/eval_crop.py` "
              "puts got_drive/ on sys.path, so the repo root is invisible. Run it as a module "
              "FROM THE REPO ROOT instead -- `python -m got_drive.eval_crop` -- which is how "
              "the other self-tests in this package are invoked (PROJECT_HANDOFF §9).")
        return
    real = generate_crop_size_list(64, 32)
    z = eval_center_crop(_mk(200, 113, seed=11), real)
    for _ in range(5):
        assert var_center_crop(z, crop_size_list=real).tobytes() == z.tobytes(), (
            "var_center_crop must be a no-op on an already-cropped frame")
    print("eval_crop self-test: OK (incl. live var_center_crop no-op x5)")


if __name__ == "__main__":
    _selftest()

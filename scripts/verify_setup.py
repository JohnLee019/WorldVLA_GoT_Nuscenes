"""Pre-flight check for nuScenes training. Run from the repo root.

Catches the failures that otherwise only surface minutes into a training run
(or worse, silently): wrong numpy/transformers pins, a missing chameleon VQGAN
tokenizer, an unresolvable ../ckpts path.

    cd ~/VLA-GoT-release && python verify_setup.py
"""
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import os
import sys
import traceback

OK, BAD = "  \033[32mok\033[0m     ", "  \033[31mFAIL\033[0m   "
results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"{OK}{name}" + (f"  ({detail})" if detail else ""))
        results.append(True)
    except Exception as e:
        print(f"{BAD}{name}\n         {type(e).__name__}: {e}")
        results.append(False)


# ------------------------------------------------------------------ versions
def v_numpy():
    import numpy
    assert numpy.__version__ == "1.26.4", (
        f"numpy is {numpy.__version__}, expected 1.26.4. "
        "torch 2.2.0 breaks on numpy 2.x ('_ARRAY_API not found'). "
        "nuscenes-devkit is the usual culprit -> pip install numpy==1.26.4"
    )
    return numpy.__version__


def v_torch():
    import torch
    assert torch.__version__.startswith("2.2.0"), f"torch is {torch.__version__}, expected 2.2.0"
    return torch.__version__


def v_transformers():
    import transformers
    assert transformers.__version__ == "4.43.0", (
        f"transformers is {transformers.__version__}, expected 4.43.0. "
        "Newer versions require torch>=2.4 and silently disable the PyTorch "
        "backend -> model classes fail to import. Fix: "
        "pip install --force-reinstall --no-deps transformers==4.43.0 tokenizers==0.19.1"
    )
    return transformers.__version__


def v_cuda():
    import torch
    assert torch.cuda.is_available(), (
        "CUDA not available. Training uses FSDP with the NCCL backend "
        "(hardcoded in xllmx/util/dist.py) and cannot run on CPU or Windows."
    )
    return f"{torch.cuda.device_count()} GPU(s), {torch.cuda.get_device_name(0)}"


# ------------------------------------------------------------------ imports
def i_model():
    from model import ChameleonXLLMXForConditionalGeneration_ck_action_head  # noqa
    return "action-head model class"


def i_solver():
    from xllmx.solvers.pretrain.pretrain_ck_action_head import PretrainSolverBase_ck_action_head  # noqa
    return "FSDP solver (no libero pulled in)"


def i_nuscenes():
    import nuscenes  # noqa
    return "nuscenes-devkit"


def i_ours():
    from data.dataset_nuscenes import NuScenesFinetuneConversation  # noqa
    from data.item_processor import FlexARItemProcessor_Action_NuScenes  # noqa
    import train_nuscenes  # noqa
    import eval_nuscenes  # noqa
    return "nuscenes dataset / item processor / train / eval"


# ------------------------------------------------------------------ assets
CKPT = os.path.abspath("../ckpts")


def a_chameleon():
    # hardcoded in item_processor.py, resolved against the CWD -> must be ../ckpts
    base = "../ckpts/chameleon/tokenizer"
    want = {"text_tokenizer.json": 1e6, "vqgan.yaml": 500, "vqgan.ckpt": 1e8}
    for f, min_size in want.items():
        p = os.path.join(base, f)
        assert os.path.isfile(p), f"missing {os.path.abspath(p)} (upload it; it is not on HuggingFace)"
        sz = os.path.getsize(p)
        assert sz >= min_size, f"{f} is only {sz} bytes -- looks truncated"
    return f"3 files under {os.path.abspath(base)}"


def a_vla():
    p = "../ckpts/VLA_model_256/libero_spatial"
    assert os.path.isdir(p), f"missing {os.path.abspath(p)}"
    assert any(f.endswith(".safetensors") for f in os.listdir(p)), "no safetensors in VLA_model_256"
    return os.path.abspath(p)


def a_lumina():
    p = "../ckpts/Lumina-mGPT-7B-768"
    assert os.path.isfile(os.path.join(p, "tokenizer.json")), f"missing tokenizer.json in {os.path.abspath(p)}"
    return os.path.abspath(p)


# ------------------------------------------------------------------ live test
def t_item_processor():
    """The real test: build the processor. This loads the VQGAN from the
    hardcoded ../ckpts path and the Lumina tokenizer, i.e. everything that
    training needs before it ever touches the model."""
    from data.item_processor import FlexARItemProcessor_Action_NuScenes
    ip = FlexARItemProcessor_Action_NuScenes(
        tokenizer="../ckpts/Lumina-mGPT-7B-768",
        target_size=256,
        device="cuda",
    )
    assert ip.token2id(ip.action_start_token) == 10004, "action_start id is not 10004"
    assert ip.token2id(ip.action_end_token) == 15004, "action_end id is not 15004"
    return "VQGAN + tokenizer loaded, action token ids 10004/15004"


def t_roundtrip():
    """waypoints -> tokens -> waypoints must recover to within half a bin."""
    import numpy as np
    from data.item_processor import FlexARItemProcessor_Action_NuScenes
    ip = FlexARItemProcessor_Action_NuScenes(
        tokenizer="../ckpts/Lumina-mGPT-7B-768", target_size=256, device="cuda",
    )
    gt = np.array([[4.5, -0.01], [9.3, 0.01], [13.5, 0.08],
                   [17.6, 0.18], [21.0, 0.26], [24.8, 0.39]])
    toks = [ip.process_action(wp)["input_ids"] for wp in gt]
    dec = np.array([ip.decode_token_ids_to_actions(np.array(t[1:-1])) for t in toks])
    rt = (dec + 1) / 2 * (ip.wp_max - ip.wp_min + 1e-8) + ip.wp_min
    half_bin = (ip.wp_max - ip.wp_min) / 255 / 2
    err = np.abs(rt - gt).max(axis=0)
    assert (err <= half_bin * 1.01).all(), f"round-trip error {err} exceeds half-bin {half_bin}"
    return f"max err x={err[0]:.4f}m y={err[1]:.4f}m (ceiling {half_bin[0]:.4f}/{half_bin[1]:.4f})"


if __name__ == "__main__":
    print("\n--- versions ---")
    check("numpy 1.26.4", v_numpy)
    check("torch 2.2.0", v_torch)
    check("transformers 4.43.0", v_transformers)
    check("CUDA", v_cuda)

    print("\n--- imports ---")
    check("model", i_model)
    check("solver", i_solver)
    check("nuscenes-devkit", i_nuscenes)
    check("our modules", i_ours)

    print("\n--- assets (../ckpts, resolved from CWD) ---")
    check("chameleon/tokenizer", a_chameleon)
    check("VLA_model_256", a_vla)
    check("Lumina tokenizer", a_lumina)

    print("\n--- live ---")
    check("build item processor", t_item_processor)
    check("waypoint round-trip", t_roundtrip)

    n_bad = results.count(False)
    print()
    if n_bad:
        print(f"\033[31m{n_bad}/{len(results)} checks FAILED -- fix these before training.\033[0m")
        sys.exit(1)
    print(f"\033[32mall {len(results)} checks passed -- ready to preprocess and train.\033[0m")

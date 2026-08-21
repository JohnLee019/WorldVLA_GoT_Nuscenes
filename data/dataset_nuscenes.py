"""
NuScenes trajectory-planning dataset for the VLA action pipeline.

Yields the same 4-tuple as `LiberoFinetuneConversation`
    (conversations, images, action, state)
so it is a drop-in for the `preprocess=='false'` (on-the-fly tokenization)
training path in pretrain_ck_action_head.py.

    - conversations : [{"from": "human", ...}, {"from": "gpt", ...}]
    - images        : list[PIL.Image]  (one per input camera)
    - action        : list of [x, y]   (future ego waypoints, metres)
    - state         : [] (unused; NuScenes planning has no proprio state here)

Input is the json produced by `data/preprocess_nuscenes.py`. `config_path`
may point directly at that json, or at a yaml with `META: {records_json: ...}`.
"""

import json
import logging
import os

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Plan a safe driving trajectory for the ego vehicle over the next 3 seconds."


class NuScenesFinetuneConversation(Dataset):
    def __init__(self, config_path, resolution=256, prompt=None,
                 with_state=False, **kwargs):
        # `kwargs` swallows with_wrist / with_action / with_world_model that the
        # base solver passes positionally-by-name for the LIBERO dataset.
        self.resolution = resolution
        self.with_state = with_state
        self.prompt = prompt or DEFAULT_PROMPT

        records_json = self._resolve_json(config_path)
        logger.info(f"[NuScenes] loading records from {records_json}")
        with open(records_json, "r") as f:
            self.data_list = json.load(f)
        logger.info(f"[NuScenes] {len(self.data_list)} records")

    @staticmethod
    def _resolve_json(config_path):
        if config_path.endswith(".json"):
            return config_path
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg["META"]["records_json"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        rec = self.data_list[idx]

        images = []
        for p in rec["images"]:
            img = Image.open(p).convert("RGB")
            images.append(img)

        # action: list of per-timestep [x, y] vectors (one <|action|> each)
        action = [list(map(float, wp)) for wp in rec["waypoints"]]

        conversations = [
            {
                "from": "human",
                "value": self.prompt + "<|image|>" * len(images),
            },
            {
                "from": "gpt",
                "value": "<|action|>" * len(action),
            },
        ]

        state = []  # keep with_state=False for the driving setup
        return conversations, images, action, state


if __name__ == "__main__":
    import sys
    ds = NuScenesFinetuneConversation(sys.argv[1])
    print("len:", len(ds))
    conv, imgs, act, st = ds[0]
    print("conversations:", conv)
    print("num images:", len(imgs), "img0 size:", imgs[0].size)
    print("action (waypoints):", np.array(act).shape, act)

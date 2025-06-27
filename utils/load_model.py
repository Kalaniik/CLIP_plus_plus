import torch
from typing import Any, Union, List
from . import build_clip


def load_clip(cfg, jit: bool = False):
    backbone_name = cfg.model_name
    url = build_clip._MODELS[backbone_name]
    
    model_path = build_clip._download(url)

    device = "cuda:{}".format(int(cfg.gpu_id)) if torch.cuda.is_available() else "cpu"

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_clip.build_model(state_dict or model.state_dict())

    return model
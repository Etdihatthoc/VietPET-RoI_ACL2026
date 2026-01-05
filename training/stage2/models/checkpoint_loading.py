import sys
from pathlib import Path
from typing import Dict, Any

import torch

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_stage1_vision_weights(model, checkpoint_path: str) -> Dict[str, Any]:
    """
    Nạp trọng số vision encoder từ checkpoint Stage 1 (MedicalCLIP).
    Chỉ copy các key bắt đầu bằng 'vision_encoder.' để tránh clash.
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("model_state_dict") or payload.get("state_dict") or payload

    vision_state = {
        k.split("vision_encoder.", 1)[1]: v
        for k, v in state_dict.items()
        if k.startswith("vision_encoder.")
    }

    if not vision_state:
        raise RuntimeError("No vision_encoder.* keys found in Stage1 checkpoint.")

    incompat = model.vision_encoder.load_state_dict(vision_state, strict=False)

    info = {
        "checkpoint": str(ckpt_path),
        "loaded_keys": len(vision_state),
        "missing": incompat.missing_keys,
        "unexpected": incompat.unexpected_keys,
    }
    return info

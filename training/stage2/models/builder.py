import sys
from pathlib import Path
from typing import Dict, Any, Tuple

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch

from hirra_model.hirra import HiRRA  # type: ignore # noqa: E402

from .checkpoint_loading import load_stage1_vision_weights
from .freeze import apply_freeze_policies, count_parameters


def build_model(config) -> Tuple[HiRRA, Dict[str, Any]]:
    model_cfg = config["model"]
    ctvit_config = dict(model_cfg["ctvit"])
    ctvit_config.pop("num_fusion_layers", None)

    model = HiRRA(
        ctvit_config=ctvit_config,
        feature_extractor_config=model_cfg["feature_extractor_config"],
        language_decoder_name=model_cfg["language_decoder_name"],
        original_spatial_dims=tuple(model_cfg.get("original_spatial_dims", (201, 160, 160))),
        num_queries=model_cfg["num_queries"],
        language_decoder_4bit=model_cfg.get("language_decoder_4bit", False),
        language_decoder_kwargs=model_cfg.get("language_decoder_kwargs"),
        use_graph_reasoning=model_cfg.get("use_graph_reasoning", True),
        graph_config=model_cfg.get("graph_config", "basic")
    )

    # Optional: toggle gradient checkpointing cho vision encoder
    model.vision_encoder.use_gradient_checkpointing = model_cfg.get("vision_gradient_checkpointing", True)

    vision_precision = model_cfg.get("vision_precision")
    if vision_precision:
        precision_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype = precision_map.get(vision_precision.lower())
        if dtype is None:
            raise ValueError(f"Unsupported vision_precision: {vision_precision}")
        model.vision_encoder = model.vision_encoder.to(dtype)

    checkpoint_info = {}
    if model_cfg.get("stage1_checkpoint"):
        checkpoint_info = load_stage1_vision_weights(model, model_cfg["stage1_checkpoint"])

    apply_freeze_policies(model, config)
    param_stats = count_parameters(model)

    build_info = {
        "checkpoint": checkpoint_info,
        "param_stats": param_stats,
    }

    return model, build_info

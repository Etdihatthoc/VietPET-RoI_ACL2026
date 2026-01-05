"""
Model Builder for Stage 3 Training
Builds HiRRA model and loads Stage 2 checkpoint.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add repo root to path
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch

from hirra_model.hirra import HiRRA  # type: ignore # noqa: E402

from .freeze import apply_stage3_freeze_policy, count_parameters, print_trainable_parameters


def _upgrade_visual_projector_norms(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map legacy VisualProjector norm weights to new roi_norm/global_norm keys.
    """
    old_w_key = "visual_projector.norm.weight"
    old_b_key = "visual_projector.norm.bias"

    if old_w_key in state_dict and old_b_key in state_dict:
        old_w = state_dict[old_w_key]
        old_b = state_dict[old_b_key]
        for new_prefix in ("visual_projector.roi_norm", "visual_projector.global_norm"):
            state_dict.setdefault(f"{new_prefix}.weight", old_w.clone())
            state_dict.setdefault(f"{new_prefix}.bias", old_b.clone())
        state_dict.pop(old_w_key, None)
        state_dict.pop(old_b_key, None)
        print("[ModelBuilder] Mapped visual_projector.norm -> roi_norm/global_norm")
    else:
        print("[ModelBuilder] No legacy visual_projector.norm found in checkpoint")

    return state_dict


def load_stage2_checkpoint(model, checkpoint_path: str) -> Dict[str, Any]:
    """
    Load full model state from Stage 2 checkpoint.

    Args:
        model: HiRRA model instance
        checkpoint_path: Path to Stage 2 best checkpoint

    Returns:
        Dictionary with loading info
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {ckpt_path}")

    print(f"[ModelBuilder] Loading Stage 2 checkpoint from: {ckpt_path}")

    # Load checkpoint
    payload = torch.load(ckpt_path, map_location="cpu")

    # Extract model state dict
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    print(f"[ModelBuilder] Original checkpoint has {len(state_dict)} keys")
    state_dict = _upgrade_visual_projector_norms(state_dict)
    model.load_state_dict(state_dict, strict=False)

    # filtered_state_dict = {
    #     k: v for k, v in state_dict.items()
    #     if not k.startswith("vision_encoder.")
    # }
    # print(f"[ModelBuilder] After filtering vision_encoder: {len(filtered_state_dict)} keys")

    # # Load into model (strict=False to allow for any architecture differences)
    # incompatible = model.load_state_dict(filtered_state_dict, strict=False)

    # Prepare info
    info = {
        "checkpoint": str(ckpt_path),
        "loaded_keys": len(state_dict),
        # "missing_keys": incompatible.missing_keys,
        # "unexpected_keys": incompatible.unexpected_keys,
        "epoch": payload.get("epoch", "unknown"),
        "global_step": payload.get("global_step", "unknown"),
    }

    # Print warnings for missing/unexpected keys
    # if incompatible.missing_keys:
    #     print(f"[Warning] Missing keys when loading checkpoint: {incompatible.missing_keys[:5]}...")
    # if incompatible.unexpected_keys:
    #     print(f"[Warning] Unexpected keys when loading checkpoint: {incompatible.unexpected_keys[:5]}...")

    print(f"[ModelBuilder] Successfully loaded {len(state_dict)} keys from Stage 2 checkpoint")
    if "epoch" in payload:
        print(f"[ModelBuilder] Checkpoint was from epoch {payload['epoch']}, step {payload.get('global_step', 'unknown')}")

    return info


def build_stage3_model(config: Dict[str, Any]) -> Tuple[HiRRA, Dict[str, Any]]:
    """
    Build HiRRA model for Stage 3 training.

    Steps:
    1. Initialize HiRRA model from config
    2. Load Stage 2 checkpoint (full model state)
    3. Apply Stage 3 freeze policy (freeze global, train ROI)
    4. Print trainable parameters summary

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, build_info)
    """
    model_cfg = config["model"]

    # Check if we have a Stage 2 checkpoint to load config from
    stage2_checkpoint = model_cfg.get("stage2_checkpoint")
    if stage2_checkpoint:
        print(f"[ModelBuilder] Loading config from Stage 2 checkpoint: {stage2_checkpoint}")
        ckpt_path = Path(stage2_checkpoint).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {ckpt_path}")

        # Load checkpoint to get config
        payload = torch.load(ckpt_path, map_location="cpu")
        if "config" in payload:
            # Use config from checkpoint for model architecture
            ckpt_model_cfg = payload["config"]["model"]
            print("[ModelBuilder] Using model config from checkpoint for architecture")

            # Use checkpoint config for architecture parameters
            ctvit_config = dict(ckpt_model_cfg.get("ctvit", {}))
            ctvit_config.pop("num_fusion_layers", None)

            feature_extractor_config = ckpt_model_cfg.get("feature_extractor_config", "improved")
            original_spatial_dims = tuple(ckpt_model_cfg.get("original_spatial_dims", (201, 480, 480)))
            num_queries = ckpt_model_cfg.get("num_queries", 32)
            use_graph_reasoning = ckpt_model_cfg.get("use_graph_reasoning", True)
            graph_config = ckpt_model_cfg.get("graph_config", "basic")
        else:
            print("[Warning] No config in checkpoint, using config from config_stage3.yaml")
            ctvit_config = dict(model_cfg.get("ctvit", {}))
            ctvit_config.pop("num_fusion_layers", None)
            feature_extractor_config = model_cfg.get("feature_extractor_config", "improved")
            original_spatial_dims = tuple(model_cfg.get("original_spatial_dims", (201, 480, 480)))
            num_queries = model_cfg.get("num_queries", 32)
            use_graph_reasoning = model_cfg.get("use_graph_reasoning", True)
            graph_config = model_cfg.get("graph_config", "basic")
    else:
        print("[ModelBuilder] No checkpoint, using config from config_stage3.yaml")
        ctvit_config = dict(model_cfg.get("ctvit", {}))
        ctvit_config.pop("num_fusion_layers", None)
        feature_extractor_config = model_cfg.get("feature_extractor_config", "improved")
        original_spatial_dims = tuple(model_cfg.get("original_spatial_dims", (201, 480, 480)))
        num_queries = model_cfg.get("num_queries", 32)
        use_graph_reasoning = model_cfg.get("use_graph_reasoning", True)
        graph_config = model_cfg.get("graph_config", "basic")

    print("[ModelBuilder] Building HiRRA model for Stage 3...")
    print(f"[ModelBuilder] CTViT config: {ctvit_config}")

    # Initialize HiRRA model with config from checkpoint (if available)
    model = HiRRA(
        ctvit_config=ctvit_config,
        feature_extractor_config=feature_extractor_config,
        language_decoder_name=model_cfg.get("language_decoder_name", "Qwen/Qwen3-8B"),
        original_spatial_dims=original_spatial_dims,
        num_queries=num_queries,
        language_decoder_4bit=model_cfg.get("language_decoder_4bit", False),
        language_decoder_kwargs=model_cfg.get("language_decoder_kwargs", {}),
        use_graph_reasoning=use_graph_reasoning,
        graph_config=graph_config
    )

    # Set gradient checkpointing for vision encoder
    use_gradient_checkpointing = model_cfg.get("vision_gradient_checkpointing", True)
    model.vision_encoder.use_gradient_checkpointing = use_gradient_checkpointing
    print(f"[ModelBuilder] Vision gradient checkpointing: {use_gradient_checkpointing}")

    # Set vision encoder precision if specified
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
        print(f"[ModelBuilder] Vision encoder precision: {vision_precision}")

    # Load Stage 2 checkpoint
    checkpoint_info = {}
    stage2_checkpoint = model_cfg.get("stage2_checkpoint")
    if stage2_checkpoint:
        checkpoint_info = load_stage2_checkpoint(model, stage2_checkpoint)
    else:
        print("[Warning] No Stage 2 checkpoint specified! Starting from scratch.")

    # Apply Stage 3 freeze policy
    print("\n[ModelBuilder] Applying Stage 3 freeze policy...")
    apply_stage3_freeze_policy(model, config)

    # Count parameters
    param_stats = count_parameters(model)

    # Print trainable parameters summary
    print_trainable_parameters(model)

    # Prepare build info
    build_info = {
        "checkpoint": checkpoint_info,
        "param_stats": param_stats,
    }

    return model, build_info


# Example usage
if __name__ == "__main__":
    import yaml

    # Test config
    test_config = {
        "model": {
            "ctvit": {
                "dim": 512,
                "patch_size": 20,
                "temporal_patch_size": 10,
                "image_size": 480,
            },
            "feature_extractor_config": "improved",
            "language_decoder_name": "Qwen/Qwen3-8B",
            "original_spatial_dims": [201, 480, 480],
            "num_queries": 32,
            "language_decoder_4bit": False,
            "use_graph_reasoning": True,
            "graph_config": "basic",
            "stage2_checkpoint": "../stage2/outputs/checkpoints/best.pt",
            "freeze": {
                "vision_encoder": True,
                "qformer": True,
                "global_projector": True,
                "fpn": True,
                "language_decoder": True,
            },
            "train": {
                "roi_extractor": True,
                "graph_module": True,
                "roi_projector": True,
            }
        }
    }

    print("Testing Stage 3 model builder...")
    print("(This is a test stub - would build actual model with config)")

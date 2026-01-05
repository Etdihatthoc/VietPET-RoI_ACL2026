"""
Model Builder for Stage 4: LoRA or full LLM finetuning.
Supports LoRA merge from Stage 3 and full LLM training.
"""

import sys
from pathlib import Path

# Add repo root to path
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch
import logging
from typing import Dict, Any, Tuple

from hirra_model.hirra import HiRRA
from .freeze import apply_stage4_freeze_policy, count_parameters
from .selective_checkpoint_loader import load_selective_checkpoints

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Warning] peft library not found. Please install: pip install peft")


def _apply_lora_for_merge(model: HiRRA, lora_cfg: Dict[str, Any]) -> None:
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required to merge LoRA. Install with: pip install peft")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        inference_mode=False,
    )

    model.language_decoder.model = get_peft_model(
        model.language_decoder.model,
        lora_config
    )


def _extract_state_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _load_roi_modules_from_stage3(model: HiRRA, checkpoint_path: str) -> Dict[str, Any]:
    stage3_path = Path(checkpoint_path).expanduser()
    if not stage3_path.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {stage3_path}")

    print(f"[ModelBuilder] Loading ROI modules from Stage 3: {stage3_path}")
    stage3_ckpt = torch.load(stage3_path, map_location="cpu")
    stage3_state = _extract_state_dict(stage3_ckpt)

    roi_prefixes = [
        'feature_extractor.roi_extractor',
        'feature_extractor.graph_builder',
        'feature_extractor.graph_reasoner',
        'feature_extractor.graph_summary_proj',
        'visual_projector.roi_projector',
        'visual_projector.roi_norm',
    ]

    roi_state = {
        k: v for k, v in stage3_state.items()
        if any(k.startswith(prefix) for prefix in roi_prefixes)
    }

    incompatible = model.load_state_dict(roi_state, strict=False)
    print(f"[ModelBuilder] ROI keys loaded: {len(roi_state)}")
    if incompatible.unexpected_keys:
        print(f"[ModelBuilder] Unexpected ROI keys: {len(incompatible.unexpected_keys)}")

    return {
        "roi_keys_loaded": len(roi_state),
        "missing_keys": len(incompatible.missing_keys) if hasattr(incompatible, 'missing_keys') else 0,
        "unexpected_keys": len(incompatible.unexpected_keys) if hasattr(incompatible, 'unexpected_keys') else 0,
    }


def load_stage3_checkpoint(model: HiRRA, checkpoint_path: str) -> Dict[str, Any]:
    """
    Load Stage 3 checkpoint into the model.

    Args:
        model: HiRRA model instance
        checkpoint_path: Path to Stage 3 checkpoint (.pt file)

    Returns:
        Dictionary with checkpoint information
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {ckpt_path}")

    print(f"[ModelBuilder] Loading Stage 3 checkpoint from: {ckpt_path}")

    # Load checkpoint
    payload = torch.load(ckpt_path, map_location="cpu")

    # Extract model state dict
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    print(f"[ModelBuilder] Checkpoint contains {len(state_dict)} keys")

    # Load into model (strict=False to allow architecture differences)
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Prepare info
    info = {
        "checkpoint": str(ckpt_path),
        "epoch": payload.get("epoch", "unknown"),
        "global_step": payload.get("global_step", "unknown"),
        "loaded_keys": len(state_dict),
        "missing_keys": len(incompatible.missing_keys) if hasattr(incompatible, 'missing_keys') else 0,
        "unexpected_keys": len(incompatible.unexpected_keys) if hasattr(incompatible, 'unexpected_keys') else 0,
    }

    # Print missing/unexpected keys if any
    if incompatible.missing_keys:
        print(f"[ModelBuilder] Missing keys: {len(incompatible.missing_keys)}")
        if len(incompatible.missing_keys) <= 10:
            for key in incompatible.missing_keys:
                print(f"  - {key}")

    if incompatible.unexpected_keys:
        print(f"[ModelBuilder] Unexpected keys: {len(incompatible.unexpected_keys)}")
        if len(incompatible.unexpected_keys) <= 10:
            for key in incompatible.unexpected_keys:
                print(f"  - {key}")

    print(f"[ModelBuilder] Successfully loaded checkpoint from epoch {info['epoch']}")

    return info


def build_stage4_model(config: Dict[str, Any]) -> Tuple[HiRRA, Dict[str, Any]]:
    """
    Build HiRRA model for Stage 4 training (LoRA finetuning).

    Steps:
    1. Load Stage 3 checkpoint to get architecture config
    2. Initialize HiRRA model with checkpoint config
    3. Load Stage 3 checkpoint weights
    4. Apply Stage 4 freeze policy (freeze all + LoRA)
    5. Count parameters

    Args:
        config: Full configuration dictionary

    Returns:
        Tuple of (model, metadata_dict)
    """
    model_cfg = config["model"]

    # Step 1: Mode selection
    warmstarted_checkpoint = model_cfg.get("warmstarted_checkpoint")
    merge_lora_from_stage2 = model_cfg.get("merge_lora_from_stage2", False)
    merge_lora_from_stage3 = model_cfg.get("merge_lora_from_stage3", False)
    load_lora_from_stage2 = model_cfg.get("load_lora_from_stage2", False)
    use_warmstarted = False

    if merge_lora_from_stage2:
        print("[ModelBuilder] Mode: merge LoRA from Stage 2 and finetune full LLM")
    elif merge_lora_from_stage3:
        print("[ModelBuilder] Mode: merge LoRA from Stage 3 and finetune full LLM")
    elif warmstarted_checkpoint:
        warmstart_path = Path(warmstarted_checkpoint).expanduser()
        if warmstart_path.exists():
            print(f"[ModelBuilder] Found warmstarted checkpoint: {warmstart_path}")
            use_warmstarted = True
        else:
            print(f"[ModelBuilder] WARNING: warmstarted_checkpoint specified but not found: {warmstart_path}")
            print("[ModelBuilder] Falling back to selective loading (Stage 2 + Stage 3)")

    # Get checkpoint paths for selective loading / merge
    stage2_checkpoint = model_cfg.get("stage2_checkpoint")
    stage3_checkpoint = model_cfg.get("stage3_checkpoint")

    if merge_lora_from_stage2:
        if not stage2_checkpoint:
            raise ValueError("stage2_checkpoint must be specified when merge_lora_from_stage2=true")
        if not stage3_checkpoint:
            raise ValueError("stage3_checkpoint must be specified to load ROI modules")
    elif merge_lora_from_stage3:
        if not stage3_checkpoint:
            raise ValueError("stage3_checkpoint must be specified when merge_lora_from_stage3=true")
    elif not use_warmstarted:
        if not stage2_checkpoint:
            raise ValueError("stage2_checkpoint must be specified in config")
        if not stage3_checkpoint:
            raise ValueError("stage3_checkpoint must be specified in config")
        if load_lora_from_stage2 and not stage2_checkpoint:
            raise ValueError("stage2_checkpoint must be specified when load_lora_from_stage2=true")

    # Load checkpoint to extract architecture config
    if merge_lora_from_stage2:
        config_ckpt_path = stage2_checkpoint
    elif merge_lora_from_stage3:
        config_ckpt_path = stage3_checkpoint
    else:
        config_ckpt_path = stage2_checkpoint
    if config_ckpt_path:
        ckpt_path = Path(config_ckpt_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for config: {ckpt_path}")
        print(f"[ModelBuilder] Loading checkpoint to extract architecture config: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
    else:
        payload = {}

    # Extract architecture config from checkpoint
    lora_cfg_for_merge = dict(model_cfg.get("lora", {}))
    lora_cfg_for_load = dict(model_cfg.get("lora", {}))

    if "config" in payload:
        ckpt_model_cfg = payload["config"]["model"]
        print("[ModelBuilder] Using model config from checkpoint for architecture")

        # Extract key architecture parameters
        ctvit_config_raw = ckpt_model_cfg.get("ctvit", {})

        # Filter ctvit_config to only include valid CTViT parameters
        # (remove MultimodalEncoder params like num_fusion_layers)
        valid_ctvit_keys = {
            'dim', 'codebook_size', 'image_size', 'patch_size', 'temporal_patch_size',
            'spatial_depth', 'temporal_depth', 'dim_head', 'heads', 'mlp_dim',
            'channels', 'use_vgg_and_gan', 'vgg_pretrained', 'discriminator_config'
        }
        ctvit_config = {k: v for k, v in ctvit_config_raw.items() if k in valid_ctvit_keys}

        feature_extractor_config = ckpt_model_cfg.get("feature_extractor_config", "improved")
        language_decoder_name = ckpt_model_cfg.get("language_decoder_name", "Qwen/Qwen3-8B")
        original_spatial_dims = tuple(ckpt_model_cfg.get("original_spatial_dims", [201, 480, 480]))
        num_queries = ckpt_model_cfg.get("num_queries", 32)
        use_graph_reasoning = ckpt_model_cfg.get("use_graph_reasoning", True)
        graph_config = ckpt_model_cfg.get("graph_config", "basic")

        # Allow config_stage4.yaml to override graph settings
        if "use_graph_reasoning" in model_cfg:
            use_graph_reasoning = model_cfg.get("use_graph_reasoning")
            print(f"[ModelBuilder] Override use_graph_reasoning from config: {use_graph_reasoning}")
        if "graph_config" in model_cfg:
            graph_config = model_cfg.get("graph_config", graph_config)
            print(f"[ModelBuilder] Override graph_config from config: {graph_config}")

        print(f"[ModelBuilder] CTViT config: {ctvit_config}")
        print(f"[ModelBuilder] Feature extractor: {feature_extractor_config}")
        print(f"[ModelBuilder] LLM: {language_decoder_name}")

        if merge_lora_from_stage2 or merge_lora_from_stage3:
            ckpt_lora_cfg = ckpt_model_cfg.get("lora")
            if ckpt_lora_cfg:
                lora_cfg_for_merge = ckpt_lora_cfg
                print("[ModelBuilder] Using LoRA config from checkpoint for merge")
        if load_lora_from_stage2:
            ckpt_lora_cfg = ckpt_model_cfg.get("lora")
            if ckpt_lora_cfg:
                lora_cfg_for_load = ckpt_lora_cfg
                print("[ModelBuilder] Using LoRA config from Stage 2 checkpoint for load")
    else:
        # Fallback: use config from config_stage4.yaml
        print("[ModelBuilder] Warning: No config in checkpoint, using config_stage4.yaml")
        ctvit_config = model_cfg.get("ctvit", {})
        feature_extractor_config = model_cfg.get("feature_extractor_config", "improved")
        language_decoder_name = model_cfg.get("language_decoder_name", "Qwen/Qwen3-8B")
        original_spatial_dims = tuple(model_cfg.get("original_spatial_dims", [201, 480, 480]))
        num_queries = model_cfg.get("num_queries", 32)
        use_graph_reasoning = model_cfg.get("use_graph_reasoning", True)
        graph_config = model_cfg.get("graph_config", "basic")

    # Step 2: Initialize HiRRA model with checkpoint architecture
    print("[ModelBuilder] Building HiRRA model for Stage 4...")
    model = HiRRA(
        ctvit_config=ctvit_config,
        feature_extractor_config=feature_extractor_config,
        language_decoder_name=language_decoder_name,
        original_spatial_dims=original_spatial_dims,
        num_queries=num_queries,
        use_graph_reasoning=use_graph_reasoning,
        graph_config=graph_config,
        language_decoder_4bit=False,  # No quantization for LoRA training
        language_decoder_kwargs={}
    )

    # Step 3: Load checkpoints
    if merge_lora_from_stage2:
        print("\n[ModelBuilder] Loading Stage 2 checkpoint (LoRA merge mode):")
        print(f"  Path: {stage2_checkpoint}")

        _apply_lora_for_merge(model, lora_cfg_for_merge)

        stage2_ckpt = torch.load(Path(stage2_checkpoint).expanduser(), map_location="cpu")
        stage2_state = _extract_state_dict(stage2_ckpt)
        model.load_state_dict(stage2_state, strict=False)

        if hasattr(model.language_decoder.model, "merge_and_unload"):
            model.language_decoder.model = model.language_decoder.model.merge_and_unload()
            print("[ModelBuilder] ✓ Merged LoRA into base LLM and unloaded adapters")
        else:
            print("[ModelBuilder] WARNING: merge_and_unload not available on LLM")

        roi_stats = _load_roi_modules_from_stage3(model, stage3_checkpoint)
        checkpoint_info = {
            "checkpoint": f"Stage2 (LoRA merged): {stage2_checkpoint} + ROI: {stage3_checkpoint}",
            "epoch": stage2_ckpt.get("epoch", "unknown"),
            "global_step": stage2_ckpt.get("global_step", "unknown"),
            "lora_merged": True,
            "roi_stats": roi_stats
        }

    elif merge_lora_from_stage3:
        print("\n[ModelBuilder] Loading Stage 3 checkpoint (LoRA merge mode):")
        print(f"  Path: {stage3_checkpoint}")

        _apply_lora_for_merge(model, lora_cfg_for_merge)
        checkpoint_info = load_stage3_checkpoint(model, stage3_checkpoint)

        if hasattr(model.language_decoder.model, "merge_and_unload"):
            model.language_decoder.model = model.language_decoder.model.merge_and_unload()
            print("[ModelBuilder] ✓ Merged LoRA into base LLM and unloaded adapters")
        else:
            print("[ModelBuilder] WARNING: merge_and_unload not available on LLM")

        checkpoint_info["lora_merged"] = True

    elif use_warmstarted:
        # Load warmstarted checkpoint (contains both Stage 2 + Stage 3 + warmstarted norms)
        print("\n[ModelBuilder] Loading WARMSTARTED checkpoint:")
        print(f"  Path: {warmstart_path}")

        warmstart_ckpt = torch.load(warmstart_path, map_location="cpu")

        if 'model_state_dict' in warmstart_ckpt:
            warmstart_state = warmstart_ckpt['model_state_dict']
        elif 'state_dict' in warmstart_ckpt:
            warmstart_state = warmstart_ckpt['state_dict']
        else:
            warmstart_state = warmstart_ckpt

        print(f"  Loading {len(warmstart_state)} keys from warmstarted checkpoint...")

        # Load with strict=False to handle any missing/extra keys
        incompatible = model.load_state_dict(warmstart_state, strict=False)

        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
            if len(incompatible.missing_keys) <= 10:
                for key in incompatible.missing_keys:
                    print(f"    - {key}")

        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
            if len(incompatible.unexpected_keys) <= 10:
                for key in incompatible.unexpected_keys:
                    print(f"    - {key}")

        checkpoint_info = {
            "checkpoint": f"Warmstarted: {warmstart_path}",
            "epoch": warmstart_ckpt.get("epoch", 0),
            "global_step": warmstart_ckpt.get("global_step", 0),
            "warmstarted": warmstart_ckpt.get("warmstarted", True),
            "warmstart_info": warmstart_ckpt.get("warmstart_info", {}),
        }

        print(f"  ✓ Loaded warmstarted checkpoint successfully!")
        if checkpoint_info.get("warmstart_info"):
            info = checkpoint_info["warmstart_info"]
            print(f"  - ROI norm warmstarted: {info.get('roi_norm_warmstarted', False)}")
            print(f"  - Global norm warmstarted: {info.get('global_norm_warmstarted', False)}")

    else:
        # Use selective loading (Stage 2 full + Stage 3 ROI only)
        print("\n[ModelBuilder] Loading checkpoints with SELECTIVE strategy:")
        print(f"  Stage 2 (global context): {stage2_checkpoint}")
        print(f"  Stage 3 (ROI modules only): {stage3_checkpoint}")

        if load_lora_from_stage2:
            print("[ModelBuilder] Pre-applying LoRA adapters to load Stage 2 LoRA weights...")
            _apply_lora_for_merge(model, lora_cfg_for_load)

        # Create a simple logger for selective loading
        logger = logging.getLogger("stage4_builder")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        checkpoint_stats = load_selective_checkpoints(
            model=model,
            stage2_checkpoint=stage2_checkpoint,
            stage3_checkpoint=stage3_checkpoint,
            device=torch.device("cpu"),  # Load on CPU first
            logger=logger
        )

        # Create checkpoint_info for metadata (use Stage 2 as primary)
        checkpoint_info = {
            "checkpoint": f"Stage2: {stage2_checkpoint} + Stage3 ROI: {stage3_checkpoint}",
            "epoch": payload.get("epoch", "unknown"),
            "global_step": payload.get("global_step", "unknown"),
            "selective_loading_stats": checkpoint_stats
        }

    # Step 4: Apply Stage 4 freeze policy (freeze all + apply LoRA)
    print("[ModelBuilder] Applying Stage 4 freeze policy (freeze all + LoRA)...")
    apply_stage4_freeze_policy(model, config)

    # Step 5: Count parameters
    param_stats = count_parameters(model)
    print(f"[ModelBuilder] Total parameters: {param_stats['total']:,}")
    print(f"[ModelBuilder] Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")

    # Return model and metadata
    metadata = {
        "checkpoint": checkpoint_info,
        "param_stats": param_stats,
        "architecture": {
            "ctvit_config": ctvit_config,
            "feature_extractor_config": feature_extractor_config,
            "language_decoder_name": language_decoder_name,
            "num_queries": num_queries,
            "use_graph_reasoning": use_graph_reasoning,
        }
    }

    return model, metadata

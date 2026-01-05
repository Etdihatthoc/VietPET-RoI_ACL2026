"""
Freeze Policies for Stage 3 Training

Stage 3: Train ROI-specific components while keeping global components frozen.

FROZEN:
- vision_encoder (from Stage 1)
- qformer/global_extractor (from Stage 2)
- global_projector (from Stage 2)
- fpn (from Stage 2)
- language_decoder (LLM stays frozen)

TRAINED:
- roi_extractor (SpatialPyramidRoI3D)
- graph_module (HierarchicalROIGraphModule)
- roi_projector
"""

from typing import Dict, Any

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Stage3][LoRA] peft chưa được cài đặt. Cài bằng: pip install peft")

def set_requires_grad(module, requires_grad: bool):
    """Set requires_grad for all parameters in a module."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_module(module):
    """Freeze a module (set requires_grad=False)."""
    set_requires_grad(module, False)


def unfreeze_module(module):
    """Unfreeze a module (set requires_grad=True)."""
    set_requires_grad(module, True)


def apply_stage3_freeze_policy(model, config: Dict[str, Any]):
    """
    Apply Stage 3 freeze policy to HiRRA model.

    Args:
        model: HiRRA model instance
        config: Configuration dict with freeze settings
    """
    freeze_cfg = config["model"].get("freeze", {})

    # 1. Freeze vision encoder (trained in Stage 1)
    if freeze_cfg.get("vision_encoder", True):
        print("[Freeze] Freezing vision_encoder")
        freeze_module(model.vision_encoder)

    # Get feature extractor and projector
    extractor = getattr(model, "feature_extractor", None)
    projector = getattr(model, "visual_projector", None)

    # 2. Freeze Q-Former/global_extractor (trained in Stage 2)
    if freeze_cfg.get("qformer", True) and extractor is not None:
        print("[Freeze] Freezing global_extractor (Q-Former)")
        freeze_module(getattr(extractor, "global_extractor", None))

    # 3. Freeze FPN (trained in Stage 2)
    if freeze_cfg.get("fpn", True) and extractor is not None:
        print("[Freeze] Freezing FPN")
        freeze_module(getattr(extractor, "fpn", None))
        freeze_module(getattr(extractor, "fpn_aggregator", None))

    # 4. Freeze global_projector (trained in Stage 2)
    if freeze_cfg.get("global_projector", True) and projector is not None:
        print("[Freeze] Freezing global_projector")
        if hasattr(projector, "global_projector"):
            freeze_module(projector.global_projector)
        if hasattr(projector, "global_norm"):
            freeze_module(projector.global_norm)

    # 5. Freeze language decoder (LLM)
    if freeze_cfg.get("language_decoder", True):
        print("[Freeze] Freezing language_decoder (LLM)")
        freeze_module(model.language_decoder)
        # Set to eval mode
        if hasattr(model.language_decoder, 'model'):
            model.language_decoder.model.eval()
        else:
            model.language_decoder.eval()

    # 6. Unfreeze ROI-specific components (to be trained in Stage 3)
    train_cfg = config["model"].get("train", {})

    if train_cfg.get("roi_extractor", True) and extractor is not None:
        print("[Train] Unfreezing roi_extractor")
        unfreeze_module(getattr(extractor, "roi_extractor", None))

    if train_cfg.get("graph_module", True) and extractor is not None:
        print("[Train] Unfreezing graph reasoning modules")
        unfreeze_module(getattr(extractor, "graph_builder", None))
        unfreeze_module(getattr(extractor, "graph_reasoner", None))
        unfreeze_module(getattr(extractor, "graph_summary_proj", None))

    if train_cfg.get("roi_projector", True) and projector is not None:
        print("[Train] Unfreezing roi_projector")
        if hasattr(projector, "roi_projector"):
            unfreeze_module(projector.roi_projector)
        if hasattr(projector, "roi_norm"):
            unfreeze_module(projector.roi_norm)

    # 7. Apply LoRA on LLM if enabled
    lora_cfg = config["model"].get("lora", {})
    if lora_cfg.get("enabled", False):
        _apply_lora(model, lora_cfg)


def count_parameters(model) -> Dict[str, float]:
    """
    Count total and trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts and percentage
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }


def print_trainable_parameters(model):
    """Print summary of trainable parameters by module."""
    print("\n" + "=" * 80)
    print("TRAINABLE PARAMETERS SUMMARY")
    print("=" * 80)

    # Overall stats
    stats = count_parameters(model)
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Trainable percentage: {stats['trainable_pct']:.2f}%")
    print("-" * 80)

    # Per-module breakdown
    modules_to_check = [
        ("vision_encoder", model.vision_encoder),
        ("feature_extractor.global_extractor", getattr(model.feature_extractor, "global_extractor", None)),
        ("feature_extractor.fpn", getattr(model.feature_extractor, "fpn", None)),
        ("feature_extractor.roi_extractor", getattr(model.feature_extractor, "roi_extractor", None)),
        ("feature_extractor.graph_builder", getattr(model.feature_extractor, "graph_builder", None)),
        ("feature_extractor.graph_reasoner", getattr(model.feature_extractor, "graph_reasoner", None)),
        ("visual_projector.global_projector", getattr(model.visual_projector, "global_projector", None)),
        ("visual_projector.roi_projector", getattr(model.visual_projector, "roi_projector", None)),
        ("language_decoder", model.language_decoder),
    ]

    for name, module in modules_to_check:
        if module is not None:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = "TRAIN" if trainable > 0 else "FROZEN"
            print(f"{name:50s} | {status:8s} | {trainable:>12,} / {total:>12,}")

    print("=" * 80 + "\n")


def _apply_lora(model, lora_cfg: Dict[str, Any]):
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required for LoRA. Install with: pip install peft")

    print("[Stage3][LoRA] Applying LoRA adapters to language decoder...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj")),
        bias="none",
        inference_mode=False,
    )

    model.language_decoder.model = get_peft_model(
        model.language_decoder.model,
        lora_config
    )
    model.language_decoder.model.train()

    if lora_cfg.get("train_embed_tokens", False):
        embed_layer = model.language_decoder.model.get_input_embeddings()
        if embed_layer is not None:
            embed_layer.weight.requires_grad = True

    if lora_cfg.get("train_lm_head", False):
        lm_head = getattr(model.language_decoder.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight.requires_grad = True

    print("[Stage3][LoRA] Trainable parameters after applying LoRA:")
    model.language_decoder.model.print_trainable_parameters()


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))

    from hirra_model.hirra import HiRRAModel

    # Mock config for Stage 3
    config = {
        "model": {
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

    # Load model (this is just for testing, would normally load from checkpoint)
    print("This is a test stub - would normally load model from checkpoint")

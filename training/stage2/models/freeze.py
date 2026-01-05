from typing import Dict, Any

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Stage2][LoRA] peft chưa được cài đặt. Cài bằng: pip install peft")


def set_requires_grad(module, requires_grad: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_module(module):
    set_requires_grad(module, False)


def apply_freeze_policies(model, config: Dict[str, Any]):
    model_cfg = config["model"]
    lora_cfg = model_cfg.get("lora", {})
    lora_enabled = lora_cfg.get("enabled", False)

    if model_cfg.get("freeze_vision_encoder", True):
        freeze_module(model.vision_encoder)

    extractor = getattr(model, "feature_extractor", None)
    projector = getattr(model, "visual_projector", None)

    if not model_cfg.get("train_fpn", False) and extractor is not None:
        freeze_module(getattr(extractor, "fpn", None))
        freeze_module(getattr(extractor, "fpn_aggregator", None))

    if model_cfg.get("freeze_roi_modules", True) and extractor is not None:
        freeze_module(getattr(extractor, "roi_extractor", None))
        freeze_module(getattr(extractor, "graph_builder", None))
        freeze_module(getattr(extractor, "graph_reasoner", None))
        freeze_module(getattr(extractor, "graph_summary_proj", None))
        if projector is not None and hasattr(projector, "roi_projector"):
            freeze_module(projector.roi_projector)

    if model_cfg.get("freeze_language_decoder", True):
        freeze_module(model.language_decoder)

    # Q-Former + global projector phải train.
    if extractor is not None:
        set_requires_grad(getattr(extractor, "global_extractor", None), True)
        if model_cfg.get("train_fpn", False):
            set_requires_grad(getattr(extractor, "fpn", None), True)
            set_requires_grad(getattr(extractor, "fpn_aggregator", None), True)

    if projector is not None and hasattr(projector, "global_projector"):
        set_requires_grad(projector.global_projector, True)

    if lora_enabled:
        _apply_lora(model, lora_cfg)
    elif model_cfg.get("freeze_language_decoder", True):
        model.language_decoder.model.eval()


def _apply_lora(model, lora_cfg: Dict[str, Any]):
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required for LoRA training. Install with `pip install peft`.")

    print("[Stage2][LoRA] Applying LoRA adapters to language decoder...")
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

    print("[Stage2][LoRA] Trainable parameters after applying LoRA:")
    model.language_decoder.model.print_trainable_parameters()


def count_parameters(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }

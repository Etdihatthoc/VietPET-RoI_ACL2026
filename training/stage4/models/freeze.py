"""
Freeze Policies for Stage 4: LoRA or full LLM finetuning.
Freeze all components except LoRA adapters or full LLM (config-driven).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Warning] peft library not found. Please install: pip install peft")


def freeze_module(module: Optional[nn.Module]) -> None:
    """
    Freeze all parameters in a module.

    Args:
        module: PyTorch module to freeze
    """
    if module is None:
        return

    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: Optional[nn.Module]) -> None:
    """
    Unfreeze all parameters in a module.

    Args:
        module: PyTorch module to unfreeze
    """
    if module is None:
        return

    for param in module.parameters():
        param.requires_grad = True


def unfreeze_lora_params(module: Optional[nn.Module]) -> None:
    """
    Unfreeze only LoRA adapter parameters in a PEFT-wrapped module.
    """
    if module is None:
        return

    for name, param in module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100.0 * trainable_params / total_params if total_params > 0 else 0.0,
    }


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print trainable parameters breakdown by component.

    Args:
        model: HiRRA model
    """
    print("\n" + "=" * 80)
    print("TRAINABLE PARAMETERS BREAKDOWN")
    print("=" * 80)

    components = {
        "Vision Encoder": getattr(model, "vision_encoder", None),
        "Feature Extractor": getattr(model, "feature_extractor", None),
        "Visual Projector": getattr(model, "visual_projector", None),
        "Language Decoder": getattr(model, "language_decoder", None),
    }

    for name, component in components.items():
        if component is None:
            continue

        stats = count_parameters(component)
        print(f"\n{name}:")
        print(f"  Total:      {stats['total']:>15,}")
        print(f"  Trainable:  {stats['trainable']:>15,} ({stats['trainable_pct']:>6.2f}%)")
        print(f"  Frozen:     {stats['frozen']:>15,}")

    # Overall stats
    overall_stats = count_parameters(model)
    print(f"\n{'Overall Model':}")
    print(f"  Total:      {overall_stats['total']:>15,}")
    print(f"  Trainable:  {overall_stats['trainable']:>15,} ({overall_stats['trainable_pct']:>6.2f}%)")
    print(f"  Frozen:     {overall_stats['frozen']:>15,}")
    print("=" * 80 + "\n")


def apply_stage4_freeze_policy(model: nn.Module, config: Dict[str, Any]) -> None:
    """
    Apply Stage 4 freeze policy: Freeze everything except LoRA adapters.

    Strategy:
    - FREEZE all components from Stages 1-3:
        - Vision encoder (Stage 1)
        - Feature extractor: FPN, Q-Former, ROI extractors, graph modules (Stages 2-3)
        - Visual projector: global and ROI projectors (Stages 2-3)
        - Base language decoder (LLM)
    - APPLY LoRA to language decoder
    - OPTIONALLY unfreeze embed_tokens and lm_head

    Args:
        model: HiRRA model
        config: Full configuration dictionary
    """
    model_cfg = config["model"]
    lora_cfg = model_cfg.get("lora", {})
    train_full_llm = model_cfg.get("train_full_llm", False)

    print("[Freeze Policy] Stage 4: Freeze all components + LoRA on LLM")

    # ========== FREEZE EVERYTHING ==========
    print("[Freeze Policy] Freezing vision encoder (Stage 1)...")
    freeze_module(model.vision_encoder)

    print("[Freeze Policy] Freezing feature extractor (Stages 2-3)...")
    freeze_module(model.feature_extractor)

    print("[Freeze Policy] Freezing visual projector (Stages 2-3)...")
    freeze_module(model.visual_projector)

    if train_full_llm:
        print("[Freeze Policy] Unfreezing full language decoder (no LoRA)...")
        unfreeze_module(model.language_decoder)
        model.language_decoder.model.train()
    else:
        print("[Freeze Policy] Freezing base language decoder...")
        freeze_module(model.language_decoder)
        # Set to eval mode to disable dropout
        model.language_decoder.model.eval()

        # ========== APPLY LoRA TO LANGUAGE DECODER ==========
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA. Install with: pip install peft")

        if lora_cfg.get("enabled", True):
            lora_already_applied = hasattr(model.language_decoder.model, "peft_config")
            if lora_already_applied:
                print("[Freeze Policy] LoRA already applied; skipping re-apply.")
                unfreeze_lora_params(model.language_decoder.model)
                if hasattr(model.language_decoder.model, "print_trainable_parameters"):
                    model.language_decoder.model.print_trainable_parameters()
            else:
                print("[Freeze Policy] Applying LoRA to language decoder...")

                # LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_cfg.get("rank", 16),
                    lora_alpha=lora_cfg.get("alpha", 32),
                    lora_dropout=lora_cfg.get("dropout", 0.05),
                    target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                    bias="none",
                    inference_mode=False,
                )

                print(f"[LoRA Config] Rank: {lora_config.r}")
                print(f"[LoRA Config] Alpha: {lora_config.lora_alpha}")
                print(f"[LoRA Config] Dropout: {lora_config.lora_dropout}")
                print(f"[LoRA Config] Target modules: {lora_config.target_modules}")

                # Apply LoRA to the LLM
                model.language_decoder.model = get_peft_model(
                    model.language_decoder.model,
                    lora_config
                )

                # Print trainable parameters from PEFT
                model.language_decoder.model.print_trainable_parameters()

    # ========== OPTIONALLY UNFREEZE EMBED_TOKENS AND LM_HEAD ==========
    if model_cfg.get("train_embed_tokens", False):
        print("[Freeze Policy] Unfreezing embed_tokens...")
        embed_layer = model.language_decoder.model.get_input_embeddings()
        if embed_layer is not None:
            embed_layer.weight.requires_grad = True
            embed_params = embed_layer.weight.numel()
            print(f"[Freeze Policy] embed_tokens: {embed_params:,} parameters trainable")

    if model_cfg.get("train_lm_head", False):
        print("[Freeze Policy] Unfreezing lm_head...")
        lm_head = model.language_decoder.model.lm_head
        if lm_head is not None:
            lm_head.weight.requires_grad = True
            lm_head_params = lm_head.weight.numel()
            print(f"[Freeze Policy] lm_head: {lm_head_params:,} parameters trainable")

    # ========== UNFREEZE VISION COMPONENTS ==========
    # New in Stage 4: Train vision components alongside LoRA
    vision_cfg = model_cfg.get("train_vision_components", {})

    # 0. Unfreeze vision encoder (CTViT)
    if vision_cfg.get("vision_encoder", False):
        print("[Freeze Policy] Unfreezing vision_encoder (CTViT)...")
        vision_encoder = getattr(model, "vision_encoder", None)
        if vision_encoder is not None:
            unfreeze_module(vision_encoder)
            ve_params = sum(p.numel() for p in vision_encoder.parameters())
            print(f"[Freeze Policy] vision_encoder: {ve_params:,} parameters trainable")

    # 1. Unfreeze roi_extractor
    if vision_cfg.get("roi_extractor", False):
        print("[Freeze Policy] Unfreezing roi_extractor...")
        roi_extractor = getattr(model.feature_extractor, "roi_extractor", None)
        if roi_extractor is not None:
            unfreeze_module(roi_extractor)
            roi_params = sum(p.numel() for p in roi_extractor.parameters())
            print(f"[Freeze Policy] roi_extractor: {roi_params:,} parameters trainable")

    # 2. Unfreeze graph_module (3 sub-components)
    if vision_cfg.get("graph_module", False):
        print("[Freeze Policy] Unfreezing graph reasoning modules...")
        graph_builder = getattr(model.feature_extractor, "graph_builder", None)
        graph_reasoner = getattr(model.feature_extractor, "graph_reasoner", None)
        graph_summary_proj = getattr(model.feature_extractor, "graph_summary_proj", None)

        total_graph_params = 0
        for module in [graph_builder, graph_reasoner, graph_summary_proj]:
            if module is not None:
                unfreeze_module(module)
                total_graph_params += sum(p.numel() for p in module.parameters())

        print(f"[Freeze Policy] graph_module: {total_graph_params:,} parameters trainable")

    # 3. Unfreeze roi_projector
    if vision_cfg.get("roi_projector", False):
        print("[Freeze Policy] Unfreezing roi_projector...")
        roi_projector = getattr(model.visual_projector, "roi_projector", None)
        if roi_projector is not None:
            unfreeze_module(roi_projector)
            roi_proj_params = sum(p.numel() for p in roi_projector.parameters())
            print(f"[Freeze Policy] roi_projector: {roi_proj_params:,} parameters trainable")

    # 4. Unfreeze global_extractor (Q-Former)
    if vision_cfg.get("global_extractor", False):
        print("[Freeze Policy] Unfreezing global_extractor (Q-Former)...")
        global_extractor = getattr(model.feature_extractor, "global_extractor", None)
        if global_extractor is not None:
            unfreeze_module(global_extractor)
            global_ext_params = sum(p.numel() for p in global_extractor.parameters())
            print(f"[Freeze Policy] global_extractor: {global_ext_params:,} parameters trainable")

    # 5. Unfreeze global_projector
    if vision_cfg.get("global_projector", False):
        print("[Freeze Policy] Unfreezing global_projector...")
        global_projector = getattr(model.visual_projector, "global_projector", None)
        if global_projector is not None:
            unfreeze_module(global_projector)
            global_proj_params = sum(p.numel() for p in global_projector.parameters())
            print(f"[Freeze Policy] global_projector: {global_proj_params:,} parameters trainable")

    # ========== PRINT PARAMETER BREAKDOWN ==========
    print_trainable_parameters(model)

    print("[Freeze Policy] Stage 4 freeze policy applied successfully!")

"""
Optimizer builder for Stage 4: LoRA or full LLM finetuning.
Creates optimizer for LoRA parameters or full LLM (+ optionally embed_tokens and lm_head).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any


def _collect_params(module: nn.Module) -> List[torch.nn.Parameter]:
    """
    Collect trainable parameters from a module.

    Args:
        module: PyTorch module

    Returns:
        List of trainable parameters
    """
    if module is None:
        return []

    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer for Stage 4 training (LoRA adapters or full LLM).

    Creates parameter groups:
    1. Full LLM parameters (if config["model"]["train_full_llm"] = True)
    2. LoRA adapters (automatically marked as trainable by PEFT)
    3. Optional: embed_tokens (if config["model"]["train_embed_tokens"] = True)
    4. Optional: lm_head (if config["model"]["train_lm_head"] = True)

    Args:
        model: HiRRA model with LoRA applied
        config: Full configuration dictionary

    Returns:
        Optimizer instance

    Raises:
        RuntimeError: If no trainable parameters found
    """
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    train_full_llm = model_cfg.get("train_full_llm", False)

    param_groups = []
    seen_params = set()  # Track parameter IDs to avoid duplicates

    # ========== DEBUG: Print all trainable parameters ==========
    print("\n[Optimizer] DEBUG: All trainable parameters in language_decoder:")
    for name, param in model.language_decoder.model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel():,} params")
    print()

    # ========== COLLECT FULL LLM (OPTIONAL) ==========
    if train_full_llm:
        llm_params = []
        for name, param in model.language_decoder.model.named_parameters():
            if param.requires_grad:
                param_id = id(param)
                if param_id not in seen_params:
                    llm_params.append(param)
                    seen_params.add(param_id)
        if llm_params:
            lr_llm = opt_cfg.get("lr_llm", 1e-5)
            param_groups.append({
                "params": llm_params,
                "lr": lr_llm,
                "name": "llm_full"
            })
            total_params = sum(p.numel() for p in llm_params)
            print(f"[Optimizer] full LLM: {total_params:,} parameters, lr={lr_llm:.2e}")

    # ========== COLLECT LORA PARAMETERS ==========
    # PEFT automatically marks LoRA adapters as trainable
    # Collect ALL trainable params from language_decoder that have "lora" in name
    if not train_full_llm:
        lora_params = []
        lora_param_names = []
        for name, param in model.language_decoder.model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                param_id = id(param)
                if param_id not in seen_params:
                    lora_params.append(param)
                    lora_param_names.append(name)
                    seen_params.add(param_id)

        if lora_params:
            lr_lora = opt_cfg.get("lr_lora", 1e-4)
            param_groups.append({
                "params": lora_params,
                "lr": lr_lora,
                "name": "lora_adapters"
            })
            print(f"[Optimizer] LoRA adapters: {len(lora_params)} parameter tensors, lr={lr_lora:.2e}")
            print(f"[Optimizer] LoRA param names: {lora_param_names[:5]}...")  # Show first 5

    # ========== COLLECT EMBED_TOKENS (OPTIONAL) ==========
    if model_cfg.get("train_embed_tokens", False):
        embed_layer = model.language_decoder.model.get_input_embeddings()
        if embed_layer is not None and embed_layer.weight.requires_grad:
            param_id = id(embed_layer.weight)
            if param_id not in seen_params:
                lr_embed = opt_cfg.get("lr_embed", 5e-5)
                param_groups.append({
                    "params": [embed_layer.weight],
                    "lr": lr_embed,
                    "name": "embed_tokens"
                })
                seen_params.add(param_id)
                print(f"[Optimizer] embed_tokens: {embed_layer.weight.numel():,} parameters, lr={lr_embed:.2e}")
            else:
                print(f"[Optimizer] Warning: embed_tokens already in another group, skipping")

    # ========== COLLECT LM_HEAD (OPTIONAL) ==========
    if model_cfg.get("train_lm_head", False):
        lm_head = model.language_decoder.model.lm_head
        if lm_head is not None and lm_head.weight.requires_grad:
            param_id = id(lm_head.weight)
            if param_id not in seen_params:
                lr_lm_head = opt_cfg.get("lr_lm_head", 5e-5)
                param_groups.append({
                    "params": [lm_head.weight],
                    "lr": lr_lm_head,
                    "name": "lm_head"
                })
                seen_params.add(param_id)
                print(f"[Optimizer] lm_head: {lm_head.weight.numel():,} parameters, lr={lr_lm_head:.2e}")
            else:
                print(f"[Optimizer] Warning: lm_head already in another group, skipping")

    # ========== COLLECT VISION COMPONENTS ==========
    # New in Stage 4: Train vision components alongside LoRA
    vision_cfg = model_cfg.get("train_vision_components", {})

    # 0. Vision Encoder (CTViT)
    if vision_cfg.get("vision_encoder", False):
        vision_encoder = getattr(model, "vision_encoder", None)
        ve_params = _collect_params(vision_encoder)

        filtered_params = [p for p in ve_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_vision_encoder = opt_cfg.get("lr_vision_encoder", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_vision_encoder,
                "name": "vision_encoder"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] vision_encoder: {total_params:,} parameters, lr={lr_vision_encoder:.2e}")

    # 1. ROI Extractor
    if vision_cfg.get("roi_extractor", False):
        roi_extractor = getattr(model.feature_extractor, "roi_extractor", None)
        roi_extractor_params = _collect_params(roi_extractor)

        filtered_params = [p for p in roi_extractor_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_roi_extractor = opt_cfg.get("lr_roi_extractor", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_roi_extractor,
                "name": "roi_extractor"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] roi_extractor: {total_params:,} parameters, lr={lr_roi_extractor:.2e}")

    # 2. Graph Module (3 sub-components combined)
    if vision_cfg.get("graph_module", False):
        graph_builder = getattr(model.feature_extractor, "graph_builder", None)
        graph_reasoner = getattr(model.feature_extractor, "graph_reasoner", None)
        graph_summary_proj = getattr(model.feature_extractor, "graph_summary_proj", None)

        graph_params = []
        for module in [graph_builder, graph_reasoner, graph_summary_proj]:
            graph_params.extend(_collect_params(module))

        filtered_params = [p for p in graph_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_graph_module = opt_cfg.get("lr_graph_module", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_graph_module,
                "name": "graph_module"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] graph_module: {total_params:,} parameters, lr={lr_graph_module:.2e}")

    # 3. ROI Projector
    if vision_cfg.get("roi_projector", False):
        roi_projector = getattr(model.visual_projector, "roi_projector", None)
        roi_projector_params = _collect_params(roi_projector)

        filtered_params = [p for p in roi_projector_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_roi_projector = opt_cfg.get("lr_roi_projector", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_roi_projector,
                "name": "roi_projector"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] roi_projector: {total_params:,} parameters, lr={lr_roi_projector:.2e}")

    # 4. Global Extractor (Q-Former)
    if vision_cfg.get("global_extractor", False):
        global_extractor = getattr(model.feature_extractor, "global_extractor", None)
        global_extractor_params = _collect_params(global_extractor)

        filtered_params = [p for p in global_extractor_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_global_extractor = opt_cfg.get("lr_global_extractor", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_global_extractor,
                "name": "global_extractor"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] global_extractor: {total_params:,} parameters, lr={lr_global_extractor:.2e}")

    # 5. Global Projector
    if vision_cfg.get("global_projector", False):
        global_projector = getattr(model.visual_projector, "global_projector", None)
        global_projector_params = _collect_params(global_projector)

        filtered_params = [p for p in global_projector_params if id(p) not in seen_params]
        for p in filtered_params:
            seen_params.add(id(p))

        if filtered_params:
            lr_global_projector = opt_cfg.get("lr_global_projector", 5e-6)
            param_groups.append({
                "params": filtered_params,
                "lr": lr_global_projector,
                "name": "global_projector"
            })
            total_params = sum(p.numel() for p in filtered_params)
            print(f"[Optimizer] global_projector: {total_params:,} parameters, lr={lr_global_projector:.2e}")

    # ========== SANITY CHECK ==========
    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freeze policy. "
            "Make sure LoRA was applied correctly and some parameters are trainable."
        )

    total_trainable_params = sum(
        sum(p.numel() for p in group["params"])
        for group in param_groups
    )
    print(f"[Optimizer] Total trainable parameters: {total_trainable_params:,}")
    print(f"[Optimizer] Number of parameter groups: {len(param_groups)}")

    # ========== CREATE OPTIMIZER ==========
    optimizer_type = opt_cfg.get("type", "AdamW")

    # Common optimizer kwargs
    optimizer_kwargs = {
        "weight_decay": opt_cfg.get("weight_decay", 0.01),
        "betas": tuple(opt_cfg.get("betas", [0.9, 0.999])),
        "eps": opt_cfg.get("eps", 1e-8),
    }

    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        print(f"[Optimizer] Using AdamW optimizer")

    elif optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, **optimizer_kwargs)
            print(f"[Optimizer] Using AdamW8bit optimizer (8-bit quantized)")
        except ImportError:
            print("[Optimizer] Warning: bitsandbytes not found, falling back to AdamW")
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print(f"[Optimizer] Weight decay: {optimizer_kwargs['weight_decay']}")
    print(f"[Optimizer] Betas: {optimizer_kwargs['betas']}")
    print(f"[Optimizer] Epsilon: {optimizer_kwargs['eps']}")

    return optimizer

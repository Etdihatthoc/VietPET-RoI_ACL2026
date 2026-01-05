from typing import Dict, List

import torch


def _collect_params(module) -> List[torch.nn.Parameter]:
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model, config: Dict):
    opt_cfg = config["optimizer"]
    param_groups = []

    qformer_params = _collect_params(getattr(model.feature_extractor, "global_extractor", None))
    if qformer_params:
        param_groups.append({
            "params": qformer_params,
            "lr": opt_cfg["lr_qformer"],
            "name": "qformer"
        })

    projector_params = _collect_params(getattr(model.visual_projector, "global_projector", None))
    if projector_params:
        param_groups.append({
            "params": projector_params,
            "lr": opt_cfg["lr_projector"],
            "name": "global_projector"
        })

    if config["model"].get("train_fpn", False):
        fpn_params = []
        fpn_params.extend(_collect_params(getattr(model.feature_extractor, "fpn", None)))
        fpn_params.extend(_collect_params(getattr(model.feature_extractor, "fpn_aggregator", None)))
        if fpn_params:
            param_groups.append({
                "params": fpn_params,
                "lr": opt_cfg.get("lr_fpn", opt_cfg["lr_qformer"]),
                "name": "fpn"
            })

    lora_module = getattr(model.language_decoder, "model", None)
    if lora_module is None:
        lora_module = getattr(model.language_decoder, "base_model", None)
    lora_params = _collect_params(lora_module)
    if lora_params:
        param_groups.append({
            "params": lora_params,
            "lr": opt_cfg.get("lr_lora", opt_cfg["lr_qformer"]),
            "name": "lora"
        })

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check freezing config.")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        betas=opt_cfg.get("betas", (0.9, 0.95))
    )

    return optimizer

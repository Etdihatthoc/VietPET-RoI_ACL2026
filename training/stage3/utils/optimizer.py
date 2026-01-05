from typing import Dict, List

import torch


def _collect_params(module) -> List[torch.nn.Parameter]:
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model, config: Dict):
    """
    Build optimizer for Stage 3 training.

    Collects trainable parameters from ROI-specific components:
    - roi_extractor (SpatialPyramidRoI3D)
    - graph modules (graph_builder, graph_reasoner, graph_summary_proj)
    - roi_projector
    """
    opt_cfg = config["optimizer"]
    param_groups = []
    seen_params = set()

    # Debug: print feature_extractor attributes
    print(f"\n[Optimizer Debug] feature_extractor attributes: {dir(model.feature_extractor)}")

    # ROI extractor parameters
    roi_extractor = getattr(model.feature_extractor, "roi_extractor", None)
    roi_extractor_params = _collect_params(roi_extractor)
    print(f"[Optimizer] roi_extractor exists: {roi_extractor is not None}, trainable params: {len(roi_extractor_params)}")
    if roi_extractor_params:
        param_groups.append({
            "params": roi_extractor_params,
            "lr": opt_cfg.get("lr_roi_extractor", 5e-5),
            "name": "roi_extractor"
        })
        for p in roi_extractor_params:
            seen_params.add(id(p))

    # Graph module parameters
    graph_builder = getattr(model.feature_extractor, "graph_builder", None)
    graph_reasoner = getattr(model.feature_extractor, "graph_reasoner", None)
    graph_summary_proj = getattr(model.feature_extractor, "graph_summary_proj", None)

    print(f"[Optimizer] graph_builder exists: {graph_builder is not None}")
    print(f"[Optimizer] graph_reasoner exists: {graph_reasoner is not None}")
    print(f"[Optimizer] graph_summary_proj exists: {graph_summary_proj is not None}")

    graph_params = []
    graph_params.extend(_collect_params(graph_builder))
    graph_params.extend(_collect_params(graph_reasoner))
    graph_params.extend(_collect_params(graph_summary_proj))
    print(f"[Optimizer] Total graph trainable params: {len(graph_params)}")

    if graph_params:
        param_groups.append({
            "params": graph_params,
            "lr": opt_cfg.get("lr_graph_module", 5e-5),
            "name": "graph_module"
        })
        for p in graph_params:
            seen_params.add(id(p))

    # ROI projector parameters
    roi_projector = getattr(model.visual_projector, "roi_projector", None)
    roi_projector_params = _collect_params(roi_projector)
    print(f"[Optimizer] roi_projector exists: {roi_projector is not None}, trainable params: {len(roi_projector_params)}")
    if roi_projector_params:
        param_groups.append({
            "params": roi_projector_params,
            "lr": opt_cfg.get("lr_roi_projector", 5e-5),
            "name": "roi_projector"
        })
        for p in roi_projector_params:
            seen_params.add(id(p))

    # LoRA / other trainable params (LLM adapters)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in seen_params]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": opt_cfg.get("lr_lora", 5e-6),
            "name": "lora_or_other"
        })
        for p in other_params:
            seen_params.add(id(p))

    print(f"\n[Optimizer] Total parameter groups: {len(param_groups)}")

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found for Stage 3. "
            "Check that ROI modules (roi_extractor, graph modules, roi_projector) are unfrozen."
        )

    # Use AdamW8bit if specified, otherwise regular AdamW
    optimizer_type = opt_cfg.get("type", "AdamW")
    if optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                betas=opt_cfg.get("betas", (0.9, 0.999)),
                eps=opt_cfg.get("eps", 1e-8)
            )
        except ImportError:
            print("[Warning] bitsandbytes not available, using regular AdamW")
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                betas=opt_cfg.get("betas", (0.9, 0.999)),
                eps=opt_cfg.get("eps", 1e-8)
            )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=opt_cfg.get("betas", (0.9, 0.999)),
            eps=opt_cfg.get("eps", 1e-8)
        )

    return optimizer

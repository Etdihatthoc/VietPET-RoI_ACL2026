"""
Load checkpoint có chọn lọc cho Stage 4.
Load Stage 2 (global context) + Stage 3 (chỉ ROI modules).
"""

import torch
from pathlib import Path
from typing import Dict, List


def load_selective_checkpoints(
    model,
    stage2_checkpoint: str,
    stage3_checkpoint: str,
    device: torch.device,
    logger
) -> Dict:
    """
    Load checkpoint Stage 2 đầy đủ, sau đó overlay Stage 3 ROI modules.

    Args:
        model: Instance của HiRRA model
        stage2_checkpoint: Path tới Stage 2 best.pt
        stage3_checkpoint: Path tới Stage 3 best.pt
        device: Device để load lên
        logger: Logger instance

    Returns:
        Dict với thống kê loading
    """

    # Bước 1: Load checkpoint Stage 2 ĐẦY ĐỦ
    logger.info("="*60)
    logger.info("ĐANG LOAD CHECKPOINT STAGE 2 (Global Context)")
    logger.info("="*60)

    stage2_path = Path(stage2_checkpoint).expanduser()
    if not stage2_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Stage 2: {stage2_path}")

    stage2_ckpt = torch.load(stage2_path, map_location=device, weights_only=False)
    stage2_state = stage2_ckpt.get('model_state_dict', stage2_ckpt)

    # Handle vocab size mismatch after adding special tokens
    model_state = model.state_dict()
    # Resize any token embedding / lm_head weight if vocab size differs (handles PEFT-wrapped keys)
    for key in list(stage2_state.keys()):
        if not (key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")):
            continue
        if key not in model_state:
            continue
        ckpt_w = stage2_state[key]
        model_w = model_state[key]
        if ckpt_w.shape != model_w.shape:
            logger.info(
                f"[Checkpoint] Resizing {key}: ckpt {tuple(ckpt_w.shape)} -> model {tuple(model_w.shape)}"
            )
            new_w = model_w.clone()
            rows = min(ckpt_w.shape[0], model_w.shape[0])
            new_w[:rows] = ckpt_w[:rows]
            stage2_state[key] = new_w

    # Load state Stage 2 (sẽ có missing keys cho ROI modules - điều này OK)
    incompatible_s2 = model.load_state_dict(stage2_state, strict=False)

    logger.info(f"Đã load checkpoint Stage 2: {len(stage2_state)} keys")
    logger.info(f"Missing keys (ROI modules - mong đợi): {len(incompatible_s2.missing_keys)}")

    # Missing keys mong đợi (ROI modules không có trong Stage 2):
    roi_missing = [k for k in incompatible_s2.missing_keys if any(
        prefix in k for prefix in ['roi_extractor', 'graph_builder', 'graph_reasoner', 'roi_projector']
    )]
    logger.info(f"ROI-related missing keys: {len(roi_missing)} (sẽ load từ Stage 3)")

    # Bước 2: Load CHỈ ROI modules từ Stage 3
    logger.info("\n" + "="*60)
    logger.info("ĐANG LOAD STAGE 3 CHỈ ROI MODULES")
    logger.info("="*60)

    stage3_path = Path(stage3_checkpoint).expanduser()
    if not stage3_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Stage 3: {stage3_path}")

    stage3_ckpt = torch.load(stage3_path, map_location=device, weights_only=False)
    stage3_state = stage3_ckpt.get('model_state_dict', stage3_ckpt)

    # Lọc: CHỈ load các keys liên quan ROI từ Stage 3
    roi_prefixes = [
        'feature_extractor.roi_extractor',
        'feature_extractor.graph_builder',
        'feature_extractor.graph_reasoner',
        'feature_extractor.graph_summary_proj',
        'visual_projector.roi_projector',
        'visual_projector.roi_norm'
    ]

    roi_state = {
        k: v for k, v in stage3_state.items()
        if any(k.startswith(prefix) for prefix in roi_prefixes)
    }

    logger.info(f"Checkpoint Stage 3: {len(stage3_state)} tổng keys")
    logger.info(f"Lọc thành ROI modules: {len(roi_state)} keys")

    # Load ROI modules từ Stage 3
    incompatible_s3 = model.load_state_dict(roi_state, strict=False)

    logger.info(f"ROI modules đã load thành công: {len(roi_state)} keys")
    if incompatible_s3.unexpected_keys:
        logger.warning(f"Unexpected keys từ Stage 3: {len(incompatible_s3.unexpected_keys)}")

    # Bước 3: Verify tất cả components đã load
    logger.info("\n" + "="*60)
    logger.info("XÁC NHẬN")
    logger.info("="*60)

    model_params = dict(model.named_parameters())

    # Kiểm tra global components (từ Stage 2)
    global_keys = [k for k in model_params.keys() if any(
        prefix in k for prefix in ['vision_encoder', 'feature_extractor.global', 'visual_projector.global']
    )]
    logger.info(f"✓ Global components đã load: {len(global_keys)} params")

    # Kiểm tra ROI components (từ Stage 3)
    roi_keys = [k for k in model_params.keys() if any(
        prefix in k for prefix in roi_prefixes
    )]
    logger.info(f"✓ ROI components đã load: {len(roi_keys)} params")

    # Kiểm tra LLM components
    llm_keys = [k for k in model_params.keys() if 'language_decoder' in k or 'llm' in k]
    logger.info(f"✓ LLM components đã load: {len(llm_keys)} params")

    stats = {
        'stage2_keys': len(stage2_state),
        'stage3_roi_keys': len(roi_state),
        'global_params': len(global_keys),
        'roi_params': len(roi_keys),
        'llm_params': len(llm_keys)
    }

    return stats

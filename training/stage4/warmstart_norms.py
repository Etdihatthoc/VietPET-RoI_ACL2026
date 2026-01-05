"""
Warmstart roi_norm và global_norm từ checkpoints Stage 2 và Stage 3.

Script này copy weights từ old 'norm' sang new 'roi_norm' và 'global_norm'
để training mượt mà hơn thay vì khởi tạo random.
"""

import torch
import argparse
from pathlib import Path


def warmstart_norms(
    stage2_checkpoint: str,
    stage3_checkpoint: str,
    output_checkpoint: str
):
    """
    Warmstart roi_norm và global_norm từ checkpoints.

    Args:
        stage2_checkpoint: Path to Stage 2 checkpoint (cho global_norm)
        stage3_checkpoint: Path to Stage 3 checkpoint (cho roi_norm)
        output_checkpoint: Path to save new checkpoint with warmstarted norms
    """
    print("=" * 80)
    print("WARMSTART NORMS FROM CHECKPOINTS")
    print("=" * 80)

    # Load Stage 3 checkpoint (có roi projector + norm)
    print(f"\n[1/4] Loading Stage 3 checkpoint: {stage3_checkpoint}")
    stage3_ckpt = torch.load(stage3_checkpoint, map_location='cpu')

    if 'model_state_dict' in stage3_ckpt:
        stage3_state = stage3_ckpt['model_state_dict']
    elif 'state_dict' in stage3_ckpt:
        stage3_state = stage3_ckpt['state_dict']
    else:
        stage3_state = stage3_ckpt

    print(f"   ✓ Loaded {len(stage3_state)} keys from Stage 3")

    # Load Stage 2 checkpoint (có global projector + norm)
    print(f"\n[2/4] Loading Stage 2 checkpoint: {stage2_checkpoint}")
    stage2_ckpt = torch.load(stage2_checkpoint, map_location='cpu')

    if 'model_state_dict' in stage2_ckpt:
        stage2_state = stage2_ckpt['model_state_dict']
    elif 'state_dict' in stage2_ckpt:
        stage2_state = stage2_ckpt['state_dict']
    else:
        stage2_state = stage2_ckpt

    print(f"   ✓ Loaded {len(stage2_state)} keys from Stage 2")

    # Tìm norm weights trong Stage 3
    print(f"\n[3/4] Extracting norm weights...")

    # Stage 3: visual_projector.norm -> roi_norm
    stage3_norm_weight = None
    stage3_norm_bias = None

    for key in stage3_state.keys():
        if 'visual_projector.norm.weight' in key:
            stage3_norm_weight = stage3_state[key].clone()
            print(f"   ✓ Found Stage 3 norm.weight: {key} {stage3_norm_weight.shape}")
        if 'visual_projector.norm.bias' in key:
            stage3_norm_bias = stage3_state[key].clone()
            print(f"   ✓ Found Stage 3 norm.bias: {key} {stage3_norm_bias.shape}")

    # Stage 2: visual_projector.norm -> global_norm
    stage2_norm_weight = None
    stage2_norm_bias = None

    for key in stage2_state.keys():
        if 'visual_projector.norm.weight' in key:
            stage2_norm_weight = stage2_state[key].clone()
            print(f"   ✓ Found Stage 2 norm.weight: {key} {stage2_norm_weight.shape}")
        if 'visual_projector.norm.bias' in key:
            stage2_norm_bias = stage2_state[key].clone()
            print(f"   ✓ Found Stage 2 norm.bias: {key} {stage2_norm_bias.shape}")

    # Kiểm tra
    if stage3_norm_weight is None or stage3_norm_bias is None:
        print("\n   ⚠️  WARNING: Could not find Stage 3 norm weights!")
        print("   Available keys in Stage 3:")
        for k in sorted([k for k in stage3_state.keys() if 'norm' in k.lower()]):
            print(f"      - {k}")

    if stage2_norm_weight is None or stage2_norm_bias is None:
        print("\n   ⚠️  WARNING: Could not find Stage 2 norm weights!")
        print("   Available keys in Stage 2:")
        for k in sorted([k for k in stage2_state.keys() if 'norm' in k.lower()]):
            print(f"      - {k}")

    # Tạo new checkpoint với warmstarted norms
    print(f"\n[4/4] Creating warmstarted checkpoint...")

    # Start from Stage 3 checkpoint (có đầy đủ ROI modules)
    new_state = {}

    # Copy Stage 2 global components
    for key, value in stage2_state.items():
        if any(prefix in key for prefix in [
            'vision_encoder.',
            'global_extractor.',
            'global_projector.',
            'fpn.',
            'language_decoder.'
        ]):
            new_state[key] = value

    # Copy Stage 3 ROI components
    for key, value in stage3_state.items():
        if any(prefix in key for prefix in [
            'roi_extractor.',
            'graph_reasoner.',
            'roi_projector.'
        ]):
            # Skip old norm weights
            if 'visual_projector.norm.' in key:
                continue
            new_state[key] = value

    # Add warmstarted norms
    if stage3_norm_weight is not None and stage3_norm_bias is not None:
        # Tìm key pattern cho visual_projector
        roi_norm_weight_key = None
        roi_norm_bias_key = None

        for key in stage3_state.keys():
            if 'visual_projector.norm.weight' in key:
                # Replace 'norm' with 'roi_norm'
                roi_norm_weight_key = key.replace('.norm.weight', '.roi_norm.weight')
                roi_norm_bias_key = key.replace('.norm.weight', '.roi_norm.bias')
                break

        if roi_norm_weight_key:
            new_state[roi_norm_weight_key] = stage3_norm_weight
            new_state[roi_norm_bias_key] = stage3_norm_bias
            print(f"   ✓ Added {roi_norm_weight_key}")
            print(f"   ✓ Added {roi_norm_bias_key}")
        else:
            print("   ⚠️  Could not determine roi_norm key pattern")

    if stage2_norm_weight is not None and stage2_norm_bias is not None:
        # Tìm key pattern cho visual_projector
        global_norm_weight_key = None
        global_norm_bias_key = None

        for key in stage2_state.keys():
            if 'visual_projector.norm.weight' in key:
                # Replace 'norm' with 'global_norm'
                global_norm_weight_key = key.replace('.norm.weight', '.global_norm.weight')
                global_norm_bias_key = key.replace('.norm.weight', '.global_norm.bias')
                break

        if global_norm_weight_key:
            new_state[global_norm_weight_key] = stage2_norm_weight
            new_state[global_norm_bias_key] = stage2_norm_bias
            print(f"   ✓ Added {global_norm_weight_key}")
            print(f"   ✓ Added {global_norm_bias_key}")
        else:
            print("   ⚠️  Could not determine global_norm key pattern")

    # Save new checkpoint
    print(f"\n[5/5] Saving warmstarted checkpoint to: {output_checkpoint}")

    # Prepare checkpoint with metadata
    output_ckpt = {
        'model_state_dict': new_state,
        'epoch': 0,
        'global_step': 0,
        'warmstarted': True,
        'warmstart_info': {
            'stage2_checkpoint': stage2_checkpoint,
            'stage3_checkpoint': stage3_checkpoint,
            'roi_norm_warmstarted': stage3_norm_weight is not None,
            'global_norm_warmstarted': stage2_norm_weight is not None,
        }
    }

    # Create output directory if needed
    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output_ckpt, output_checkpoint)
    print(f"   ✓ Saved {len(new_state)} keys to checkpoint")

    print("\n" + "=" * 80)
    print("WARMSTART COMPLETE!")
    print("=" * 80)
    print(f"\nNew checkpoint: {output_checkpoint}")
    print(f"Total keys: {len(new_state)}")
    print(f"ROI norm warmstarted: {stage3_norm_weight is not None}")
    print(f"Global norm warmstarted: {stage2_norm_weight is not None}")
    print("\nYou can now use this checkpoint for Stage 4 training with:")
    print(f"  model.load_state_dict(torch.load('{output_checkpoint}')['model_state_dict'], strict=False)")


def main():
    parser = argparse.ArgumentParser(description="Warmstart roi_norm and global_norm from checkpoints")
    parser.add_argument(
        "--stage2",
        type=str,
        default="training/stage2/outputs_dataAThai/checkpoints/best.pt",
        help="Path to Stage 2 checkpoint"
    )
    parser.add_argument(
        "--stage3",
        type=str,
        default="training/stage3/outputs_ROI_Lora_reviewed/checkpoints/best.pt",
        help="Path to Stage 3 checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/stage4/warmstarted_checkpoint.pt",
        help="Path to save warmstarted checkpoint"
    )

    args = parser.parse_args()

    warmstart_norms(
        stage2_checkpoint=args.stage2,
        stage3_checkpoint=args.stage3,
        output_checkpoint=args.output
    )


if __name__ == "__main__":
    main()

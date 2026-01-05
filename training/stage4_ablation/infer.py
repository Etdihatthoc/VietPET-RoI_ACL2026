"""
Stage 4 Inference Script
Load checkpoint and generate predictions with ROI reasoning capability.
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from torch.utils.data import DataLoader
from training.stage4_ablation.data import Stage3Dataset, Stage3Collator
from training.stage4_ablation.models.builder import build_stage4_model
from training.stage4_ablation.engine.evaluation import generate_predictions, export_predictions_to_csv
from training.stage4_ablation.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 Inference - Generate predictions with ROI reasoning")
    parser.add_argument("--config", type=str, default="config_stage4.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--output", type=str, default="./inference_predictions.csv", help="Output CSV path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate (default: None = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--max_gen_tokens", type=int, default=1024, help="Max tokens to generate per sample")
    return parser.parse_args()


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str, base_dir: Path):
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).expanduser()
    return path.expanduser().resolve()


def build_dataloaders(config: dict, tokenizer):
    """Build train and validation DataLoaders (for inference we only use val)."""
    data_cfg = config["data"]
    prompt_cfg = config["prompt"]
    system_cfg = config["system"]

    # Prefer explicit val_json; fallback to legacy json_path
    val_json = data_cfg.get("val_json") or data_cfg.get("json_path")
    if val_json is None:
        raise KeyError("No val_json or json_path found in config['data']")

    # Create val dataset only (no further split)
    val_dataset = Stage3Dataset(
        json_path=val_json,
        data_root=data_cfg["data_root"],
        split="val",
        train_split=None,
        seed=data_cfg.get("seed", 42)
    )

    # Create collator
    collator = Stage3Collator(tokenizer, prompt_cfg)

    # Create val DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 2),
        shuffle=False,  # Don't shuffle for inference
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=system_cfg.get("pin_memory", True),
        persistent_workers=False
    )

    return None, val_loader  # Return None for train_loader


def load_checkpoint_for_inference(model, checkpoint_path: str, device):
    """Load checkpoint weights for inference (model only, no optimizer/scheduler)."""
    print(f"[Inference] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load with strict=False to handle LoRA adapters
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print("[Warning] Missing keys (may be normal for LoRA):")
        print(f"  {incompatible.missing_keys[:5]}...")  # Show first 5
    if incompatible.unexpected_keys:
        print("[Warning] Unexpected keys:")
        print(f"  {incompatible.unexpected_keys[:5]}...")  # Show first 5

    model = model.to(device)
    model.eval()

    print(f"[Inference] ✓ Checkpoint loaded successfully")
    if 'epoch' in checkpoint:
        print(f"[Inference] ✓ Checkpoint was from epoch {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"[Inference] ✓ Global step: {checkpoint['global_step']}")

    return model


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent

    # Resolve paths
    data_cfg = config["data"]
    for key in ["json_path", "train_json", "val_json"]:
        if key in data_cfg:
            data_cfg[key] = str(resolve_path(data_cfg[key], base_dir))
    if "data_root" in data_cfg:
        data_cfg["data_root"] = str(resolve_path(data_cfg["data_root"], base_dir))

    model_cfg = config["model"]
    if model_cfg.get("stage3_checkpoint"):
        model_cfg["stage3_checkpoint"] = str(resolve_path(model_cfg["stage3_checkpoint"], base_dir))

    set_seed(config["system"]["seed"])

    print("=" * 70)
    print("[Stage 4 Inference]")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples if args.num_samples else 'ALL'}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Build model
    print("\n[1/4] Building model...")
    model, _ = build_stage4_model(config)  # Unpack tuple (model, build_info)
    print(f"✓ Model built")

    # Load checkpoint for inference
    print(f"\n[2/4] Loading checkpoint...")
    device = torch.device(args.device)
    model = load_checkpoint_for_inference(model, args.checkpoint, device)
    tokenizer = model.get_tokenizer()

    # Build dataloader (we only need val_loader)
    print(f"\n[3/4] Building dataloaders...")
    _, val_loader = build_dataloaders(config, tokenizer)
    print(f"✓ Validation batches: {len(val_loader)}")

    # Generate predictions
    print(f"\n[4/4] Generating predictions...")
    print(f"  Max generation tokens: {args.max_gen_tokens}")
    print(f"  Temperature: 0.1")

    predictions = generate_predictions(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,  # CRITICAL FIX: Pass torch.device object, not string
        max_new_tokens=args.max_gen_tokens,
        temperature=0.1,  # Moderate temperature to avoid early EOS
        top_p=0.9
    )

    # Filter to num_samples if specified (None means all)
    total_generated = len(predictions)
    if args.num_samples is not None and args.num_samples < total_generated:
        predictions = predictions[:args.num_samples]
        print(f"  Filtered to first {args.num_samples} samples (from {total_generated} total)")

    # Export to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Export] Saving to: {output_path}")
    export_predictions_to_csv(predictions, str(output_path))

    print("\n" + "=" * 70)
    print("[SUCCESS] Inference completed!")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Output file: {output_path}")
    print("=" * 70)

    # Print first 2 predictions as sample
    print("\n" + "=" * 70)
    print("Sample Predictions:")
    print("=" * 70)
    for i, pred in enumerate(predictions[:2]):
        print(f"\n[Sample {i+1}]")
        print(f"  Patient ID: {pred['patient_id']}")
        print(f"  Region: {pred['region']}")
        print(f"  Num ROIs: {pred['num_rois']}")
        print(f"  Ground Truth (first 200 chars):")
        print(f"    {pred['ground_truth'][:200]}...")
        print(f"  Prediction (first 200 chars):")
        print(f"    {pred['prediction'][:200]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()

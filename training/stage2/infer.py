"""
Stage 2 Inference Script
Load checkpoint and generate predictions for a subset of validation samples.
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

from training.stage2.data import build_dataloaders
from training.stage2.models import build_model
from training.stage2.engine.evaluation import generate_predictions, export_predictions_to_csv
from training.stage2.utils import (
    setup_logger,
    set_seed,
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Inference - Generate predictions from checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--output", type=str, default="./inference_predictions.csv", help="Output CSV path")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate (default: 20)")
    parser.add_argument("--device", type=str, default=None, help="torch device override (e.g., cuda:0).")
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


def load_checkpoint_for_inference(model, checkpoint_path: str, device):
    """Load checkpoint weights for inference (model only, no optimizer/scheduler)."""
    print(f"[Inference] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[Warning] Checkpoint loading had some incompatibilities:")
        if incompatible.missing_keys:
            print(f"  Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {incompatible.unexpected_keys}")

    model = model.to(device)
    model.eval()

    print(f"[Inference] ✓ Checkpoint loaded successfully")
    if 'epoch' in checkpoint:
        print(f"[Inference] ✓ Checkpoint was from epoch {checkpoint['epoch']}")

    return model


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent

    # Resolve important paths relative to config
    data_cfg = config["data"]
    data_cfg["dataset_root"] = str(resolve_path(data_cfg["dataset_root"], base_dir))

    model_cfg = config["model"]
    if model_cfg.get("stage1_checkpoint"):
        model_cfg["stage1_checkpoint"] = str(resolve_path(model_cfg["stage1_checkpoint"], base_dir))

    set_seed(config["system"]["seed"])

    logger = setup_logger("./inference_logs")
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model, build_info = build_model(config)
    logger.info(
        f"Trainable params: {build_info['param_stats']['trainable'] / 1e6:.2f}M "
        f"({build_info['param_stats']['trainable_pct']:.2f}%) "
        f"| Total params: {build_info['param_stats']['total'] / 1e6:.2f}M"
    )

    # Load checkpoint for inference
    model = load_checkpoint_for_inference(model, args.checkpoint, device)
    tokenizer = model.get_tokenizer()

    # Build dataloader (we only need val_loader)
    logger.info("Building dataloaders...")
    _, val_loader = build_dataloaders(config, tokenizer)
    logger.info(f"Validation batches: {len(val_loader)}")

    # Generate predictions
    logger.info(f"Generating predictions for {args.num_samples} samples...")
    logger.info(f"Max generation tokens: {args.max_gen_tokens}")

    predictions = generate_predictions(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_gen_tokens,
        temperature=0.5,  # Moderate temperature to avoid early EOS
        top_p=0.9,
        max_samples=args.num_samples
    )

    # Export to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting predictions to: {output_path}")
    export_predictions_to_csv(predictions, str(output_path))

    logger.info("=" * 70)
    logger.info("Inference completed!")
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 70)

    # Print first 2 predictions as sample
    print("\n" + "=" * 70)
    print("Sample Predictions:")
    print("=" * 70)
    for i, pred in enumerate(predictions[:2]):
        print(f"\n[Sample {i+1}]")
        print(f"Patient ID: {pred['patient_id']}")
        print(f"Region: {pred['region']}")
        print(f"Ground Truth: {pred['ground_truth'][:200]}...")
        print(f"Prediction: {pred['prediction'][:200]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()

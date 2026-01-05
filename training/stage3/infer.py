"""
Stage 3 Inference Script (ROI-only)
Load Stage 3 checkpoint and generate ROI descriptions on the val split.
Outputs:
  - CSV of predictions (for inspection)
  - JSON with predicted ROI descriptions merged into val JSON (for Stage 4)
"""

import argparse
import json
import sys
from pathlib import Path
import re

import yaml
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from training.stage3.data import Stage3Dataset, Stage3Collator
from training.stage3.models.builder import build_stage3_model
from training.stage3.engine.evaluation import generate_predictions, export_predictions_to_csv
from training.stage3.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 Inference - Generate ROI descriptions on val set")
    parser.add_argument("--config", type=str, default="config_stage3.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 3 checkpoint (.pt)")
    parser.add_argument("--output_csv", type=str, default="./stage3_inference.csv", help="Output CSV path")
    parser.add_argument("--output_json", type=str, default="./stage3_val_with_pred.json", help="Output JSON with predicted ROI descriptions merged")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate (default: all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_gen_tokens", type=int, default=512, help="Max tokens to generate per sample")
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


def build_dataloader(config: dict, tokenizer):
    """Build val DataLoader for Stage 3."""
    data_cfg = config["data"]
    prompt_cfg = config["prompt"]
    system_cfg = config["system"]

    val_json = data_cfg.get("val_json") or data_cfg.get("json_path")
    if val_json is None:
        raise KeyError("No val_json or json_path found in config['data']")

    val_dataset = Stage3Dataset(
        json_path=val_json,
        data_root=data_cfg["data_root"],
        split="val",
        train_split=None,
        seed=data_cfg.get("seed", 42)
    )

    collator = Stage3Collator(tokenizer, prompt_cfg)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 2),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=system_cfg.get("pin_memory", True),
        persistent_workers=False
    )
    return val_loader


def load_checkpoint_for_inference(model, checkpoint_path: str, device):
    print(f"[Inference] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"[Warning] Missing keys (first 5): {incompatible.missing_keys[:5]}")
    if incompatible.unexpected_keys:
        print(f"[Warning] Unexpected keys (first 5): {incompatible.unexpected_keys[:5]}")

    model = model.to(device)
    model.eval()
    print("[Inference] ✓ Checkpoint loaded")
    return model


def parse_roi_descriptions(text: str) -> list[str]:
    """Parse generated text into per-ROI descriptions."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    roi_desc = []
    for ln in lines:
        m = re.match(r"roi\\s*\\d+\\s*:\\s*(.*)", ln, flags=re.IGNORECASE)
        if m:
            roi_desc.append(m.group(1).strip())
        else:
            # fallback: if line not prefixed ROI, keep as-is
            roi_desc.append(ln)
    return roi_desc


def merge_predictions_into_json(val_json_path: Path, predictions: list[dict], output_json: Path):
    """Merge predicted ROI descriptions into val JSON as 'pred_description' per ROI."""
    data = json.loads(val_json_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])

    # Build lookup by (patient_id, region)
    pred_map = {(p["patient_id"], p["region"]): p for p in predictions}

    updated = 0
    for sample in samples:
        key = (sample.get("patient_id"), sample.get("region"))
        pred_entry = pred_map.get(key)
        if not pred_entry:
            continue
        roi_list = parse_roi_descriptions(pred_entry["prediction"])
        for idx, roi in enumerate(sample.get("rois", [])):
            if idx < len(roi_list):
                roi["pred_description"] = roi_list[idx]
        updated += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Export] Merged predictions into JSON: {output_json} (samples updated: {updated})")


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
    if model_cfg.get("stage2_checkpoint"):
        model_cfg["stage2_checkpoint"] = str(resolve_path(model_cfg["stage2_checkpoint"], base_dir))

    set_seed(config["system"]["seed"])

    print("=" * 70)
    print("[Stage 3 Inference - ROI descriptions]")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output JSON: {args.output_json}")
    print(f"Samples: {args.num_samples if args.num_samples else 'ALL'}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Build model
    print("\n[1/4] Building model...")
    model, _ = build_stage3_model(config)
    print("✓ Model built")

    # Load checkpoint
    print("\n[2/4] Loading checkpoint...")
    device = torch.device(args.device)
    model = load_checkpoint_for_inference(model, args.checkpoint, device)
    tokenizer = model.get_tokenizer()

    # Build dataloader
    print("\n[3/4] Building dataloader (val)...")
    val_loader = build_dataloader(config, tokenizer)
    print(f"✓ Validation batches: {len(val_loader)}")

    # Generate predictions
    print("\n[4/4] Generating predictions...")
    predictions = generate_predictions(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_gen_tokens,
        temperature=0.5,
        top_p=0.9
    )

    # Filter to num_samples if specified
    if args.num_samples is not None:
        predictions = predictions[:args.num_samples]
        print(f"Filtered to first {args.num_samples} samples")

    # Export CSV
    export_predictions_to_csv(predictions, args.output_csv)

    # Merge into val JSON
    val_json_path = Path(data_cfg.get("val_json") or data_cfg.get("json_path"))
    merge_predictions_into_json(val_json_path, predictions, Path(args.output_json))

    # Print sample
    if predictions:
        print("\nSample prediction:")
        print(f"Patient: {predictions[0]['patient_id']} | Region: {predictions[0]['region']}")
        print(f"Pred (first 200 chars): {predictions[0]['prediction'][:200]}...")


if __name__ == "__main__":
    main()

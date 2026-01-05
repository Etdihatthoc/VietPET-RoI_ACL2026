"""
Dump a CSV with patient_id, region, num_rois, and decoded prompt input + target for Stage 4 to inspect batching.
"""
import argparse
import csv
from pathlib import Path
import sys
import yaml
import torch

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from torch.utils.data import DataLoader
from training.stage4.data import Stage3Dataset, Stage3Collator
from training.stage4.models.builder import build_stage4_model
from training.stage4.utils import set_seed


def resolve_path(path_str, base_dir: Path):
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).expanduser()
    return path.expanduser().resolve()


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_val_loader(config: dict, tokenizer):
    data_cfg = config["data"]
    prompt_cfg = config["prompt"]
    system_cfg = config["system"]

    val_json = data_cfg.get("val_json") or data_cfg.get("json_path")
    if val_json is None:
        raise KeyError("No val_json or json_path in config['data']")

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
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=system_cfg.get("pin_memory", True),
        persistent_workers=False,
    )
    return val_loader


def main():
    parser = argparse.ArgumentParser(description="Dump Stage4 val prompts to CSV")
    parser.add_argument("--config", default="config_stage4.yaml", help="Path to config_stage4.yaml")
    parser.add_argument("--output", default="./stage4_inference_inputs.csv", help="Output CSV path")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent

    # Resolve paths
    data_cfg = config["data"]
    for key in ["json_path", "train_json", "val_json"]:
        if key in data_cfg:
            data_cfg[key] = str(resolve_path(data_cfg[key], base_dir))
    data_cfg["data_root"] = str(resolve_path(data_cfg["data_root"], base_dir))

    model_cfg = config["model"]
    if model_cfg.get("stage3_checkpoint"):
        model_cfg["stage3_checkpoint"] = str(resolve_path(model_cfg["stage3_checkpoint"], base_dir))

    set_seed(config["system"]["seed"])

    # Build model to get tokenizer
    model, _ = build_stage4_model(config)
    tokenizer = model.get_tokenizer()

    val_loader = build_val_loader(config, tokenizer)

    rows = []
    for idx, batch in enumerate(val_loader):
        prompt_ids = batch["input_ids"][0]
        prompt_mask = batch["attention_mask"][0]
        target_ids = batch["target_ids"][0]
        target_mask = batch["target_mask"][0]

        prompt_text = tokenizer.decode(prompt_ids[prompt_mask.bool()], skip_special_tokens=False)
        target_text = tokenizer.decode(target_ids[target_mask.bool()], skip_special_tokens=False)

        rows.append({
            "patient_id": batch["patient_ids"][0],
            "region": batch["regions"][0],
            "num_rois": batch["num_rois"][0],
            "input": prompt_text,
            "target": target_text,
        })
        if idx >= 20:  # only first 5 samples
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "num_rois", "input", "target"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

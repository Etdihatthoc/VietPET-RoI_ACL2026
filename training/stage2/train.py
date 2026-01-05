import argparse
import sys
from pathlib import Path

import yaml

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch

from training.stage2.data import build_dataloaders
from training.stage2.models import build_model
from training.stage2.engine import build_optimizer, build_scheduler, Stage2Trainer
from training.stage2.utils import (
    setup_logger,
    init_wandb,
    set_seed,
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Training - Global Context Alignment")
    parser.add_argument("--config", type=str, default="config_stage2.yaml", help="Path to config YAML.")
    parser.add_argument("--device", type=str, default=None, help="torch device override (e.g., cuda:0).")
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

    train_cfg = config["training"]
    train_cfg["checkpoint_dir"] = str(resolve_path(train_cfg["checkpoint_dir"], base_dir))
    train_cfg["log_dir"] = str(resolve_path(train_cfg["log_dir"], base_dir))
    if train_cfg.get("resume_from"):
        train_cfg["resume_from"] = str(resolve_path(train_cfg["resume_from"], base_dir))

    set_seed(config["system"]["seed"])

    logger = setup_logger(config["training"]["log_dir"])
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    model, build_info = build_model(config)
    ckpt_meta = build_info.get("checkpoint", {})
    if ckpt_meta:
        logger.info(f"Loaded Stage1 checkpoint: {ckpt_meta.get('checkpoint')}")
    else:
        logger.info("Stage1 checkpoint not provided.")
    logger.info(
        f"Trainable params: {build_info['param_stats']['trainable'] / 1e6:.2f}M "
        f"({build_info['param_stats']['trainable_pct']:.2f}%)"
        f" | Total params: {build_info['param_stats']['total'] / 1e6:.2f}M"
    )

    model = model.to(device)
    tokenizer = model.get_tokenizer()

    train_loader, val_loader = build_dataloaders(config, tokenizer)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    wandb_run = init_wandb(config)

    trainer = Stage2Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger,
        wandb_run=wandb_run
    )
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

"""
Stage 3 Training Script: Local (RoI) Feature Alignment

Trains ROI-specific components:
- SpatialPyramidRoI3D (ROI extractor)
- HierarchicalROIGraphModule (Graph reasoning)
- roi_projector (ROI projection head)

All other components (vision encoder, Q-Former, global projector, LLM) are frozen.
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add repo root to path
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data import Stage3Dataset, Stage3Collator
from models import build_stage3_model
from engine import Stage3Trainer
from utils import (
    setup_logger,
    init_wandb,
    set_seed,
    get_device,
    build_optimizer,
    build_scheduler,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 3 Training - Local (RoI) Feature Alignment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_stage3.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., 'cuda:0', 'cpu')"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str, base_dir: Path):
    """Resolve path relative to base directory."""
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).expanduser()
    return path.expanduser().resolve()


def build_dataloaders(config: dict, tokenizer) -> tuple:
    """
    Build train and validation DataLoaders.

    Args:
        config: Configuration dictionary
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = config["data"]
    prompt_cfg = config["prompt"]
    system_cfg = config["system"]

    # Create datasets (prefer explicit train/val json if provided)
    train_json = data_cfg.get("train_json", data_cfg.get("json_path"))
    val_json = data_cfg.get("val_json", data_cfg.get("json_path"))
    train_split = data_cfg.get("train_split", 0.8) if train_json == val_json else None

    train_dataset = Stage3Dataset(
        json_path=train_json,
        data_root=data_cfg["data_root"],
        split="train",
        train_split=train_split,
        seed=data_cfg.get("seed", 42)
    )

    val_dataset = Stage3Dataset(
        json_path=val_json,
        data_root=data_cfg["data_root"],
        split="val",
        train_split=train_split,
        seed=data_cfg.get("seed", 42)
    )

    # Create collator
    collator = Stage3Collator(tokenizer, prompt_cfg)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collator,
        pin_memory=system_cfg.get("pin_memory", True),
        persistent_workers=system_cfg.get("persistent_workers", False) and data_cfg.get("num_workers", 2) > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collator,
        pin_memory=system_cfg.get("pin_memory", True),
        persistent_workers=system_cfg.get("persistent_workers", False) and data_cfg.get("num_workers", 2) > 0
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    config = load_config(config_path)
    base_dir = config_path.parent

    # Resolve paths relative to config file
    data_cfg = config["data"]
    # Resolve data_root
    if "data_root" in data_cfg:
        data_cfg["data_root"] = str(resolve_path(data_cfg["data_root"], base_dir))

    # Resolve train/val jsons if provided
    if "train_json" in data_cfg:
        data_cfg["train_json"] = str(resolve_path(data_cfg["train_json"], base_dir))
    if "val_json" in data_cfg:
        data_cfg["val_json"] = str(resolve_path(data_cfg["val_json"], base_dir))
    # Fallback for legacy single json_path
    if "json_path" in data_cfg:
        data_cfg["json_path"] = str(resolve_path(data_cfg["json_path"], base_dir))

    model_cfg = config["model"]
    if model_cfg.get("stage2_checkpoint"):
        model_cfg["stage2_checkpoint"] = str(resolve_path(model_cfg["stage2_checkpoint"], base_dir))

    output_cfg = config.get("output", {})
    for key in ["checkpoints_dir", "predictions_dir", "logs_dir"]:
        if key in output_cfg:
            output_cfg[key] = str(resolve_path(output_cfg[key], base_dir))

    train_cfg = config["training"]
    if train_cfg.get("resume_from"):
        train_cfg["resume_from"] = str(resolve_path(train_cfg["resume_from"], base_dir))

    # Set seed for reproducibility
    set_seed(config["system"]["seed"])

    # Setup logger
    logger = setup_logger(output_cfg.get("logs_dir", "outputs/logs"), name="stage3")
    logger.info("=" * 80)
    logger.info("STAGE 3 TRAINING: LOCAL (RoI) FEATURE ALIGNMENT")
    logger.info("=" * 80)

    # Get device
    device = get_device(args.device or config["system"]["device"])
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING MODEL")
    logger.info("=" * 80)
    model, build_info = build_stage3_model(config)

    # Log model info
    ckpt_meta = build_info.get("checkpoint", {})
    if ckpt_meta:
        logger.info(f"Loaded Stage 2 checkpoint: {ckpt_meta.get('checkpoint')}")
        logger.info(f"  - From epoch: {ckpt_meta.get('epoch', 'unknown')}")
        logger.info(f"  - Global step: {ckpt_meta.get('global_step', 'unknown')}")
    else:
        logger.warning("Stage 2 checkpoint not provided! Training from scratch.")

    param_stats = build_info["param_stats"]
    logger.info(f"Total parameters: {param_stats['total'] / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {param_stats['trainable'] / 1e6:.2f}M ({param_stats['trainable_pct']:.2f}%)")

    # Move model to device
    model = model.to(device)

    # Get tokenizer
    tokenizer = model.get_tokenizer()
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Build dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING DATALOADERS")
    logger.info("=" * 80)
    train_loader, val_loader = build_dataloaders(config, tokenizer)
    logger.info(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    logger.info(f"Val batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")

    # Build optimizer
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING OPTIMIZER & SCHEDULER")
    logger.info("=" * 80)
    optimizer = build_optimizer(model, config)
    logger.info(f"Optimizer: {config['optimizer']['type']}")
    logger.info(f"Learning rates:")
    for group in optimizer.param_groups:
        logger.info(f"  - {group.get('name', 'default')}: {group['lr']:.2e}")

    # Build scheduler
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    logger.info(f"Scheduler: {config['scheduler']['type']}")

    # Initialize WandB
    wandb_run = None
    if config.get("wandb", {}).get("enabled", False):
        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZING WANDB")
        logger.info("=" * 80)
        wandb_run = init_wandb(config)
        logger.info(f"WandB run: {config['wandb']['project']}/{config['wandb']['run_name']}")

    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAINER")
    logger.info("=" * 80)
    trainer = Stage3Trainer(
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

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

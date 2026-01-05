from pathlib import Path
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "config": config,
        "extra": extra or {}
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu"
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "extra": checkpoint.get("extra", {})
    }

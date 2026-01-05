import logging
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(log_dir: str, name: str = "stage2"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def init_wandb(config: Dict[str, Any]):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("WandB requested but not installed.") from exc

    print("[Wandb] Logging in...")
    wandb.login(key=wandb_cfg['api_key'],relogin=True)
    wandb.login(key=wandb_cfg['api_key'],relogin=True)
    run = wandb.init(
        project=wandb_cfg["project"],
        name=wandb_cfg.get("name", "stage2"),

        config=config,
        reinit=True
    )
    return run


def log_to_wandb(run, metrics: Dict[str, Any], step: Optional[int] = None):
    if run is None:
        return
    run.log(metrics, step=step)

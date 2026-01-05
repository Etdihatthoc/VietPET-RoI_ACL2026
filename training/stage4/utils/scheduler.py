from typing import Dict

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def build_scheduler(optimizer, config: Dict, steps_per_epoch: int):
    train_cfg = config["training"]
    total_epochs = train_cfg["epochs"]
    total_steps = max(1, steps_per_epoch * total_epochs)

    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    warmup_steps = int(total_steps * warmup_ratio)

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_steps
            )
        )
        milestones.append(warmup_steps)

    cosine_steps = max(1, total_steps - warmup_steps)
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=train_cfg.get("eta_min", 1e-6)
        )
    )
    if not milestones:
        return schedulers[0]

    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones
    )

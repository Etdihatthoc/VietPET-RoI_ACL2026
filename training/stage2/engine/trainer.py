from pathlib import Path
from typing import Dict, Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .evaluation import summarize_validation, generate_predictions, export_predictions_to_csv
from ..utils import save_checkpoint, load_checkpoint, log_to_wandb


class Stage2Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        tokenizer,
        config: Dict[str, Any],
        device: torch.device,
        logger,
        wandb_run=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = logger
        self.wandb_run = wandb_run

        self.train_cfg = config["training"]
        self.grad_accum = self.train_cfg["grad_accum"]
        self.mixed_precision = config["system"].get("mixed_precision", True) and device.type == "cuda"
        self.autocast_dtype = torch.bfloat16

        self.checkpoint_dir = Path(self.train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_dir = Path(self.train_cfg.get("predictions_dir", "./outputs/predictions"))
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        resume_path = self.train_cfg.get("resume_from")
        if resume_path:
            self._resume(Path(resume_path))

    def train(self):
        num_epochs = self.train_cfg["epochs"]
        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self._train_one_epoch(epoch)

            metrics = {"train/loss": train_loss, "epoch": epoch + 1}
            log_to_wandb(self.wandb_run, metrics, step=self.global_step)
            self.logger.info(f"[Epoch {epoch + 1}] Train loss: {train_loss:.4f}")

            if (epoch + 1) % self.train_cfg.get("validation_interval", 1) == 0:
                val_loss = self.validate(epoch)
                val_metrics = summarize_validation(val_loss)
                log_to_wandb(
                    self.wandb_run,
                    {
                        "val/loss": val_metrics["loss"],
                        "val/perplexity": val_metrics["perplexity"],
                        "epoch": epoch + 1
                    },
                    step=self.global_step
                )
                self.logger.info(
                    f"[Epoch {epoch + 1}] Val loss: {val_metrics['loss']:.4f} "
                    f"PPL: {val_metrics['perplexity']:.2f}"
                )
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self._save("best.pt", epoch + 1)

            self._save("last.pt", epoch + 1)

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        steps = 0
        accum_step_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for step, batch in enumerate(pbar):
            loss = self._forward_batch(batch)
            raw_loss = loss.detach()
            accum_step_loss += raw_loss.item()
            loss = loss / self.grad_accum
            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                clip_grad_norm_(self.model.parameters(), self.train_cfg["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1

                step_loss = accum_step_loss / self.grad_accum
                accum_step_loss = 0.0
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.global_step % 10 == 0:
                    log_to_wandb(
                        self.wandb_run,
                        {
                            "train/step_loss": step_loss,
                            "train/lr": current_lr,
                            "train/global_step": self.global_step
                        },
                        step=self.global_step
                    )

            loss_value = loss.item()
            running_loss += loss_value
            steps += 1

            if step % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": loss_value, "lr": f"{current_lr:.2e}"})

        avg_loss = running_loss / max(1, steps)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int = None):
        """
        Validate the model and export predictions to CSV.

        Args:
            epoch: Current epoch number (for CSV filename)

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            loss = self._forward_batch(batch, backward=False)
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)

        # Generate predictions and export to CSV
        # Only generate for specific epochs (save time)
        generate_csv_epochs = self.train_cfg.get("generate_csv_epochs", None)
        should_generate = False

        if epoch is not None:
            if generate_csv_epochs is None:
                # If not specified, generate for all epochs
                should_generate = True
            else:
                # Only generate for specified epochs
                should_generate = (epoch + 1) in generate_csv_epochs

        if should_generate:
            self.logger.info(f"[Epoch {epoch + 1}] Generating predictions for CSV export...")
            try:
                predictions = generate_predictions(
                    model=self.model,
                    dataloader=self.val_loader,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_new_tokens=self.train_cfg.get("max_gen_tokens", 512),
                    temperature=0.5,  # Moderate temperature to avoid early EOS
                    top_p=0.9,
                    max_samples=self.train_cfg.get("val_subset_size", None)
                )

                # Export to CSV
                csv_path = self.predictions_dir / f"epoch_{epoch + 1}_predictions.csv"
                export_predictions_to_csv(predictions, str(csv_path))
                self.logger.info(f"[Epoch {epoch + 1}] Predictions exported to {csv_path}")

            except Exception as e:
                self.logger.warning(f"[Epoch {epoch + 1}] Failed to generate/export predictions: {e}")
        else:
            self.logger.info(f"[Epoch {epoch + 1}] Skipping CSV generation (not in generate_csv_epochs list)")

        return avg_loss

    def _forward_batch(self, batch, backward: bool = True):
        ct = batch["ct"].unsqueeze(1).to(self.device, dtype=torch.float32, non_blocking=True)
        pet = batch["pet"].unsqueeze(1).to(self.device, dtype=torch.float32, non_blocking=True)
        boxes_list = [b.to(self.device, dtype=torch.float32) for b in batch["boxes_list"]]
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        target_ids = batch["target_ids"].to(self.device, non_blocking=True)

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision
        )

        with autocast_ctx:
            outputs = self.model(
                ct_image=ct,
                pet_image=pet,
                boxes_list=boxes_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids
            )
            loss = outputs.loss

        if not backward:
            return loss.detach()
        return loss

    def _save(self, filename: str, epoch: int):
        path = self.checkpoint_dir / filename
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            config=self.config,
            extra={"tokenizer_len": len(self.tokenizer)}
        )

    def _resume(self, checkpoint_path: Path):
        # Check if only loading weights (recommended when changing batch_size/grad_accum)
        weights_only = self.train_cfg.get("resume_weights_only", False)

        if weights_only:
            # Only load model weights, reset optimizer/scheduler
            info = load_checkpoint(
                checkpoint_path,
                self.model,
                optimizer=None,  # Don't load optimizer
                scheduler=None,  # Don't load scheduler
                map_location=self.device
            )
            self.logger.info(
                f"⚠️  Loaded ONLY model weights from {checkpoint_path} "
                f"(optimizer & scheduler RESET due to config changes)"
            )
            self.logger.info("    → Training will start from epoch 0 with fresh optimizer state")
        else:
            # Load full training state (may cause issues if batch_size/grad_accum changed!)
            info = load_checkpoint(
                checkpoint_path,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device
            )
            self.start_epoch = info["epoch"]
            self.global_step = info["global_step"]
            self.best_val_loss = info["best_val_loss"]
            self.logger.info(
                f"Resumed full training state from {checkpoint_path} "
                f"(epoch={self.start_epoch}, global_step={self.global_step})"
            )

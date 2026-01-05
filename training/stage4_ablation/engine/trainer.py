"""
Stage 3 Trainer
Training loop for Stage 3 with ROI-specific components.
"""

from pathlib import Path
from typing import Dict, Any
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .evaluation import compute_metrics, export_predictions_to_csv, generate_predictions
from utils import save_checkpoint, load_checkpoint, log_to_wandb


class Stage3Trainer:
    """
    Trainer for Stage 3: Local (RoI) Feature Alignment.

    Features:
    - Gradient accumulation
    - Mixed precision training
    - Validation with CSV export of predictions
    - Checkpoint saving every epoch
    - WandB logging
    """

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
        """
        Args:
            model: HiRRA model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            tokenizer: HuggingFace tokenizer
            config: Configuration dictionary
            device: Device to train on
            logger: Logger instance
            wandb_run: WandB run instance (optional)
        """
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

        # Training config
        self.train_cfg = config["training"]
        self.grad_accum = self.train_cfg.get("grad_accum", 1)
        self.max_grad_norm = self.train_cfg.get("max_grad_norm", 1.0)
        self.label_smoothing = float(self.train_cfg.get("label_smoothing", 0.0))
        if self.label_smoothing > 0:
            setattr(self.model, "label_smoothing", self.label_smoothing)
            self.logger.info(f"[Trainer] Label smoothing enabled: {self.label_smoothing}")

        # Mixed precision
        mixed_precision = self.train_cfg.get("mixed_precision", "bf16")
        self.use_mixed_precision = mixed_precision and device.type == "cuda"
        if self.use_mixed_precision:
            if mixed_precision == "bf16" or mixed_precision == "bfloat16":
                self.autocast_dtype = torch.bfloat16
            elif mixed_precision == "fp16" or mixed_precision == "float16":
                self.autocast_dtype = torch.float16
            else:
                self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32

        # Output directories
        output_cfg = config.get("output", {})
        self.checkpoint_dir = Path(output_cfg.get("checkpoints_dir", "outputs/checkpoints"))
        self.predictions_dir = Path(output_cfg.get("predictions_dir", "outputs/predictions"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # PHASE 1 FIX: Early stopping configuration
        early_stop_cfg = self.train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = early_stop_cfg.get("patience") is not None
        if self.early_stopping_enabled:
            self.patience = early_stop_cfg.get("patience", 5)
            self.min_delta = early_stop_cfg.get("min_delta", 0.0)
            self.patience_counter = 0
            self.logger.info(f"[Early Stopping] Enabled with patience={self.patience}, min_delta={self.min_delta}")
        else:
            self.logger.info("[Early Stopping] Disabled")

        # Resume from checkpoint if specified
        resume_path = self.train_cfg.get("resume_from")
        if resume_path:
            self._resume(Path(resume_path))

        self.logger.info(f"[Trainer] Initialized Stage3Trainer")
        self.logger.info(f"[Trainer] Mixed precision: {self.use_mixed_precision} ({self.autocast_dtype})")
        self.logger.info(f"[Trainer] Gradient accumulation: {self.grad_accum}")
        self.logger.info(f"[Trainer] Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"[Trainer] Predictions dir: {self.predictions_dir}")

    def train(self):
        """Main training loop."""
        num_epochs = self.train_cfg["epochs"]
        save_every_epoch = self.train_cfg.get("save_every_epoch", True)
        validation_every_epoch = self.train_cfg.get("validation_every_epoch", True)

        self.logger.info(f"[Trainer] Starting training for {num_epochs} epochs")
        self.logger.info(f"[Trainer] Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"[Trainer] Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.start_epoch, num_epochs):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            self.logger.info(f"{'='*80}")

            # Train one epoch
            train_loss = self._train_one_epoch(epoch)

            # Log training metrics
            metrics = {"train/loss": train_loss, "epoch": epoch + 1}
            log_to_wandb(self.wandb_run, metrics, step=self.global_step)
            self.logger.info(f"[Epoch {epoch + 1}] Train loss: {train_loss:.4f}")

            # Validation
            if validation_every_epoch:
                val_loss = self.validate(epoch)
                val_metrics = compute_metrics(val_loss)

                # Log validation metrics
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
                    f"[Epoch {epoch + 1}] Val loss: {val_metrics['loss']:.4f}, "
                    f"Perplexity: {val_metrics['perplexity']:.2f}"
                )

                # Save best checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.logger.info(f"[Epoch {epoch + 1}] New best validation loss: {val_loss:.4f}")
                    self.best_val_loss = val_loss
                    self._save("best.pt", epoch + 1)

                # PHASE 1 FIX: Early stopping check
                if self.early_stopping_enabled:
                    if is_best:
                        self.patience_counter = 0  # Reset counter when validation improves
                        self.logger.info(f"[Early Stopping] Validation improved, resetting patience counter")
                    else:
                        # Check if improvement is significant enough
                        improvement = self.best_val_loss - val_loss
                        if improvement < self.min_delta:
                            self.patience_counter += 1
                            self.logger.info(
                                f"[Early Stopping] No significant improvement. "
                                f"Patience: {self.patience_counter}/{self.patience}"
                            )

                            if self.patience_counter >= self.patience:
                                self.logger.info(
                                    f"[Early Stopping] Stopping training at epoch {epoch + 1}. "
                                    f"Best val_loss: {self.best_val_loss:.4f} at epoch {epoch + 1 - self.patience}"
                                )
                                break  # Exit training loop
                        else:
                            self.logger.info(
                                f"[Early Stopping] Small improvement ({improvement:.4f}), "
                                f"resetting patience counter"
                            )
                            self.patience_counter = 0

            # Save checkpoint every epoch
            if save_every_epoch:
                self._save(f"epoch_{epoch + 1}.pt", epoch + 1)

            # Always save last checkpoint
            self._save("last.pt", epoch + 1)

        self.logger.info(f"\n{'='*80}")
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"{'='*80}\n")

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

        for step, batch in enumerate(pbar):
            # Forward pass
            loss = self._forward_batch(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum
            loss.backward()

            # Update weights after accumulation
            if (step + 1) % self.grad_accum == 0:
                # Clip gradients
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Log to wandb every N steps
                if self.global_step % self.train_cfg.get("log_interval", 10) == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    log_to_wandb(
                        self.wandb_run,
                        {
                            "train/step_loss": loss.item() * self.grad_accum,
                            "train/lr": current_lr,
                        },
                        step=self.global_step
                    )

            # Update running loss
            loss_value = loss.item()
            running_loss += loss_value
            steps += 1

            # Update progress bar
            if step % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{current_lr:.2e}"
                })

        avg_loss = running_loss / max(1, steps)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Validate the model and export predictions to CSV.

        Args:
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        steps = 0

        # For predictions export
        all_predictions = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

        for batch in pbar:
            # Forward pass
            loss = self._forward_batch(batch, backward=False)
            total_loss += loss.item()
            steps += 1

            # Collect predictions for CSV export
            # (We'll do a separate generation pass for this)

        avg_loss = total_loss / max(1, steps)

        # Generate predictions for CSV export every 5 epochs (epoch 0,5,10,...)
        if epoch + 1 == 20:
            self.logger.info(f"[Epoch {epoch + 1}] Generating predictions for CSV export...")
            try:
                predictions = generate_predictions(
                    model=self.model,
                    dataloader=self.val_loader,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_new_tokens=self.train_cfg.get("max_gen_tokens", 512),
                    temperature=0.1,
                    top_p=0.9
                )

                # Export to CSV
                csv_path = self.predictions_dir / f"epoch_{epoch + 1}_predictions.csv"
                export_predictions_to_csv(predictions, str(csv_path))

            except Exception as e:
                self.logger.warning(f"[Epoch {epoch + 1}] Failed to generate/export predictions: {e}")

        return avg_loss

    def _forward_batch(self, batch: Dict[str, Any], backward: bool = True) -> torch.Tensor:
        """
        Forward pass on a batch.

        Args:
            batch: Batch dictionary from DataLoader
            backward: Whether this is for backward pass (training)

        Returns:
            Loss tensor
        """
        # Move batch to device
        ct = batch["ct"].to(self.device, dtype=torch.float32, non_blocking=True)
        pet = batch["pet"].to(self.device, dtype=torch.float32, non_blocking=True)
        boxes_list = [boxes.to(self.device, dtype=torch.float32) for boxes in batch["boxes_list"]]
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        target_ids = batch["target_ids"].to(self.device, non_blocking=True)

        # Autocast context for mixed precision
        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_mixed_precision
        )

        with autocast_ctx:
            # Forward pass through model
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
        """
        Save checkpoint.

        Args:
            filename: Checkpoint filename
            epoch: Current epoch number
        """
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
        self.logger.info(f"[Checkpoint] Saved to {path}")

    def _resume(self, checkpoint_path: Path):
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"[Resume] Loading checkpoint from {checkpoint_path}")

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
            f"[Resume] Resumed from epoch {self.start_epoch}, "
            f"global_step {self.global_step}, "
            f"best_val_loss {self.best_val_loss:.4f}"
        )

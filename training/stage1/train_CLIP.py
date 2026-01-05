"""
Stage 1 Training Script with Wandb Logging

Features:
- CLIP-style contrastive learning
- Layered learning rates (vision vs text)
- Cosine LR scheduler with warmup
- Mixed precision training (bf16)
- Gradient accumulation
- Wandb real-time tracking
- Checkpoint saving
- Simple retrieval metrics (R@1)
"""

import os
import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import argparse
import wandb
from dataset_v2 import create_dataloaders
#from dataset import create_dataloaders
from model import MedicalCLIP, clip_loss
import bitsandbytes as bnb

class Trainer:
    """Main trainer class với Wandb integration."""

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Trainer] Using device: {self.device}")

        # ==================== Wandb Login & Init ====================
        if config['wandb']['enabled']:
            print("[Wandb] Logging in...")
            wandb.login(
                key=config['wandb']['api_key'],
                relogin=True
            )

            print("[Wandb] Initializing run...")
            wandb.init(
                project=config['wandb']['project'],
                name=config['wandb']['name'],
                config=config
            )

        # ==================== Model ====================
        print("[Trainer] Creating model...")
        self.model = MedicalCLIP(config)
        
        # Load pretrained weights if specified
        

        # Move model to device (handle 4-bit text encoder)
        if config['model'].get('load_text_in_4bit', False):
            # Text encoder đã ở GPU qua device_map="auto"
            # PHẢI GÁN LẠI kết quả .to() vì PyTorch .to() KHÔNG in-place!
            print("[Trainer] Đang chuyển vision encoder lên GPU...")

            # CÁCH ĐƠN GIẢN NHẤT: Move toàn bộ vision_encoder một lần
            # .to() trả về module MỚI, PHẢI gán lại!
            self.model.vision_encoder = self.model.vision_encoder.to(self.device)
            self.model.vision_proj = self.model.vision_proj.to(self.device)
            self.model.text_proj = self.model.text_proj.to(self.device)

            # Move logit_scale parameter (dùng .data để giữ Parameter wrapper)
            self.model.logit_scale.data = self.model.logit_scale.data.to(self.device)

            # Verify placement (kiểm tra THẬT SỰ đã lên GPU chưa)
            ct_device = next(self.model.vision_encoder.ct_encoder.parameters()).device
            pet_device = next(self.model.vision_encoder.pet_encoder.parameters()).device
            text_device = next(self.model.text_encoder.parameters()).device
            vision_proj_device = next(self.model.vision_proj.parameters()).device

            print(f"[Trainer] ✓ CT encoder trên: {ct_device}")
            print(f"[Trainer] ✓ PET encoder trên: {pet_device}")
            print(f"[Trainer] ✓ Text encoder trên: {text_device}")
            print(f"[Trainer] ✓ Vision projection trên: {vision_proj_device}")
            print(f"[Trainer] ✓ Logit scale trên: {self.model.logit_scale.device}")

            # Kiểm tra GPU memory để đảm bảo model đã lên GPU
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                mem_reserve = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[GPU] Bộ nhớ đã dùng: {mem_alloc:.2f} GB")
                print(f"[GPU] Bộ nhớ đã cấp phát: {mem_reserve:.2f} GB")

                # CẢNH BÁO nếu memory quá thấp
                if mem_alloc < 1.0:
                    print("[CẢNH BÁO] ⚠️  GPU memory < 1GB - Model có thể vẫn ở CPU!")
                    print("[CẢNH BÁO] ⚠️  Hãy kiểm tra lại device placement!")
        else:
            # Move entire model to device (không dùng 4-bit)
            self.model = self.model.to(self.device)
            print(f"[Trainer] ✓ Model đã chuyển lên: {self.device}")

        # Wandb watch model (track gradients and parameters)
        # DISABLED: log='all' causes severe performance degradation
        # if config['wandb']['enabled']:
        #     wandb.watch(self.model, log='gradients', log_freq=100)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params / 1e6:.1f}M")
        print(f"[Model] Trainable params: {trainable_params / 1e6:.1f}M")

        # ==================== Data ====================
        print("[Trainer] Creating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(config)

        # ==================== Optimizer ====================
        print("[Trainer] Creating optimizer...")

        # Separate param groups với layered LRs
        # IMPORTANT: Only include trainable parameters (filter frozen ones)
        vision_params = [p for p in self.model.vision_encoder.parameters() if p.requires_grad]
        vision_params += [p for p in self.model.vision_proj.parameters() if p.requires_grad]

        text_params = [p for p in self.model.text_encoder.parameters() if p.requires_grad]
        text_params += [p for p in self.model.text_proj.parameters() if p.requires_grad]

        # Count trainable params per group
        vision_trainable = sum(p.numel() for p in vision_params)
        text_trainable = sum(p.numel() for p in text_params)
        print(f"[Optimizer] Vision trainable params: {vision_trainable / 1e6:.1f}M")
        print(f"[Optimizer] Text trainable params: {text_trainable / 1e6:.1f}M")

        # self.optimizer = AdamW([
        #     {
        #         'params': vision_params,
        #         'lr': config['training']['lr_vision'],
        #         'name': 'vision'
        #     },
        #     {
        #         'params': text_params,
        #         'lr': config['training']['lr_text'],
        #         'name': 'text'
        #     },
        #     {
        #         'params': [self.model.logit_scale],
        #         'lr': config['training']['lr_vision'],
        #         'name': 'temperature'
        #     }
        # ], weight_decay=config['training']['weight_decay'])
        
        # Build optimizer param groups
        param_groups = [
            {'params': vision_params, 'lr': config['training']['lr_vision'], 'name': 'vision'},
            {'params': [self.model.logit_scale], 'lr': config['training']['lr_vision'], 'name': 'temperature'}
        ]

        # Only add text params if there are any trainable (PhoBERT might be frozen)
        if len(text_params) > 0:
            param_groups.append({'params': text_params, 'lr': config['training']['lr_text'], 'name': 'text'})
            print(f"[Optimizer] Text params INCLUDED (unfrozen)")
        else:
            print(f"[Optimizer] Text params EXCLUDED (all frozen)")

        self.optimizer = bnb.optim.AdamW8bit(
            param_groups,
            weight_decay=config['training']['weight_decay']
        )
        
        print(f"[Optimizer] Vision LR: {config['training']['lr_vision']:.2e}")
        print(f"[Optimizer] Text LR: {config['training']['lr_text']:.2e}")

        # ==================== LR Scheduler ====================
        # Warmup + Cosine annealing
        total_steps = config['training']['epochs'] * len(self.train_loader)
        warmup_steps = config['training']['warmup_epochs'] * len(self.train_loader)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        print(f"[Scheduler] Warmup steps: {warmup_steps}")
        print(f"[Scheduler] Total steps: {total_steps}")

        # ==================== Mixed Precision ====================
        # Note: GradScaler is NOT needed for BFloat16 (only for Float16)
        # BFloat16 has the same dynamic range as FP32, so no scaling needed
        self.scaler = None
        if config['mixed_precision']:
            #self.scaler = torch.amp.GradScaler()
            print("[Trainer] Mixed precision (bf16) enabled - GradScaler disabled for BFloat16")

        # ==================== State ====================
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # Create checkpoint directory
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

        # ==================== Resume (optional) ====================
        resume_path = self.config['training'].get('resume_from')
        if resume_path:
            resume_path = os.path.expanduser(resume_path)
            self.load_checkpoint(resume_path)

    def train_epoch(self):
        """Train một epoch."""
        self.model.train()
        total_loss = 0
        grad_accum = self.config['training']['grad_accum']

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.config['training']['epochs']}")

        for step, batch in enumerate(pbar):
            # Move to device - FORCE to CUDA
            ct = batch['ct'].to(self.device, non_blocking=True)
            pet = batch['pet'].to(self.device, non_blocking=True)
            texts = batch['text']

            # Debug: Print device for first batch
            if step == 0:
                print(f"[Debug] CT on device: {ct.device}")
                print(f"[Debug] PET on device: {pet.device}")
                print(f"[Debug] CT shape: {ct.shape}")  # Should be [B, 201, 480, 480]
                print(f"[Debug] PET shape: {pet.shape}")  # Should be [B, 201, 480, 480]
                assert ct.shape[1] == 201, f"Expected depth 201, got {ct.shape[1]}"
                assert (ct.shape[1] - 1) % 10 == 0, f"(depth-1) must be divisible by 10"

            # Forward pass với mixed precision
            with torch.amp.autocast(
                device_type='cuda',
                enabled=self.config['mixed_precision'],
                dtype=torch.bfloat16
            ):
                image_embeds, text_embeds, temp = self.model(ct, pet, texts)
                loss = clip_loss(image_embeds, text_embeds, temp)
                loss = loss / grad_accum  # Scale loss for gradient accumulation

            # ✅ NaN detection - Stop immediately if loss diverges
            if not torch.isfinite(loss):
                print(f"\n[ERROR] ❌ Loss is NaN/Inf at epoch {self.epoch}, step {step}")
                print(f"  Temperature: {temp.item():.4f}")
                print(f"  Image embeds: min={image_embeds.min():.4f}, max={image_embeds.max():.4f}")
                print(f"  Text embeds: min={text_embeds.min():.4f}, max={text_embeds.max():.4f}")
                raise ValueError("Training diverged - loss became NaN/Inf")

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient checking (first batch of first epoch)
            if self.epoch == 1 and step == 0:
                print("\n[Gradient Check] Verifying gradient flow...")
                grad_info = []

                # Check vision encoder
                vision_grads = [p.grad for p in self.model.vision_encoder.parameters() if p.grad is not None]
                if vision_grads:
                    avg_grad = torch.stack([g.abs().mean() for g in vision_grads]).mean().item()
                    grad_info.append(f"Vision encoder: {len(vision_grads)} params with grad (avg: {avg_grad:.2e})")
                else:
                    grad_info.append("Vision encoder: NO GRADIENTS!")

                # Check text encoder
                text_grads = [p.grad for p in self.model.text_encoder.parameters() if p.grad is not None]
                if text_grads:
                    avg_grad = torch.stack([g.abs().mean() for g in text_grads]).mean().item()
                    grad_info.append(f"Text encoder: {len(text_grads)} params with grad (avg: {avg_grad:.2e})")
                else:
                    grad_info.append("Text encoder: FROZEN (expected if freeze_phobert=true)")

                # Check projection heads
                vision_proj_grads = [p.grad for p in self.model.vision_proj.parameters() if p.grad is not None]
                text_proj_grads = [p.grad for p in self.model.text_proj.parameters() if p.grad is not None]
                if vision_proj_grads:
                    avg_grad = torch.stack([g.abs().mean() for g in vision_proj_grads]).mean().item()
                    grad_info.append(f"Vision proj: {len(vision_proj_grads)} params (avg: {avg_grad:.2e})")
                if text_proj_grads:
                    avg_grad = torch.stack([g.abs().mean() for g in text_proj_grads]).mean().item()
                    grad_info.append(f"Text proj: {len(text_proj_grads)} params (avg: {avg_grad:.2e})")

                # Check logit_scale
                if self.model.logit_scale.grad is not None:
                    grad_info.append(f"Logit scale: grad={self.model.logit_scale.grad.item():.2e}")
                else:
                    grad_info.append("Logit scale: NO GRADIENT!")

                print("\n".join(grad_info))
                print("[Gradient Check] Done!\n")

            # if step % 10 == 0:  # Clear mỗi 10 steps
            #     torch.cuda.empty_cache()

            # Optimizer step (mỗi grad_accum steps)
            if (step + 1) % grad_accum == 0:
                # Unscale gradients and clip
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # ✅ Check for NaN/Inf gradients before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

                if not torch.isfinite(total_norm):
                    print(f"\n[ERROR] ❌ Gradient is NaN/Inf at epoch {self.epoch}, step {step}")
                    print(f"  Total gradient norm: {total_norm.item()}")
                    raise ValueError("Training diverged - gradients became NaN/Inf")

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Wandb logging (step-level) - Log mỗi 50 steps để tránh network bottleneck
                if self.config['wandb']['enabled'] and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss.item() * grad_accum,
                        'train/temperature': temp.item(),
                        'train/lr_vision': self.optimizer.param_groups[0]['lr'],
                        'train/lr_text': self.optimizer.param_groups[1]['lr'],
                        'train/global_step': self.global_step
                    })

            # Update progress bar (no print to avoid I/O blocking)
            total_loss += loss.item() * grad_accum
            pbar.set_postfix({
                'loss': f"{loss.item() * grad_accum:.4f}",
                'temp': f"{temp.item():.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = total_loss / len(self.train_loader)

        # Wandb epoch-level logging
        if self.config['wandb']['enabled']:
            wandb.log({
                'train/epoch_loss': avg_loss,
                'epoch': self.epoch
            })

        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validation với simple retrieval metrics."""
        self.model.eval()
        total_loss = 0

        # Collect all embeddings
        all_image_embeds = []
        all_text_embeds = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            ct = batch['ct'].to(self.device, non_blocking=True)
            pet = batch['pet'].to(self.device, non_blocking=True)
            texts = batch['text']

            with torch.amp.autocast(
                device_type='cuda',
                enabled=self.config['mixed_precision'],
                dtype=torch.bfloat16
            ):
                image_embeds, text_embeds, temp = self.model(ct, pet, texts)
                loss = clip_loss(image_embeds, text_embeds, temp)

            total_loss += loss.item()
            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())

        avg_loss = total_loss / len(self.val_loader)

        # Compute retrieval metrics
        all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
        all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [N, D]

        # Image-to-Text retrieval (R@1)
        sim_i2t = all_image_embeds @ all_text_embeds.T  # [N, N]
        ranks_i2t = sim_i2t.argsort(dim=1, descending=True)
        correct_i2t = (ranks_i2t[:, 0] == torch.arange(len(ranks_i2t))).float().mean()

        # Text-to-Image retrieval (R@1)
        sim_t2i = all_text_embeds @ all_image_embeds.T  # [N, N]
        ranks_t2i = sim_t2i.argsort(dim=1, descending=True)
        correct_t2i = (ranks_t2i[:, 0] == torch.arange(len(ranks_t2i))).float().mean()

        # Wandb logging
        if self.config['wandb']['enabled']:
            wandb.log({
                'val/loss': avg_loss,
                'val/recall@1_i2t': correct_i2t.item(),
                'val/recall@1_t2i': correct_t2i.item(),
                'epoch': self.epoch
            })

        return avg_loss, correct_i2t.item(), correct_t2i.item()

    def save_checkpoint(self, filename):
        """Save checkpoint."""
        path = os.path.join(self.config['training']['checkpoint_dir'], filename)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint: {path}")

        # Wandb artifact
        if self.config['wandb']['enabled']:
            artifact = wandb.Artifact(
                name=f"model-epoch{self.epoch}",
                type='model',
                metadata={
                    'epoch': self.epoch,
                    'val_loss': self.best_val_loss,
                    'global_step': self.global_step
                }
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def load_checkpoint(self, path):
        """Resume full training state from a checkpoint."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[Trainer] Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location='cpu')

        incompat = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            print("[Trainer] Warning: checkpoint/model mismatch detected")
            if incompat.missing_keys:
                print(f"  Missing keys: {incompat.missing_keys}")
            if incompat.unexpected_keys:
                print(f"  Unexpected keys: {incompat.unexpected_keys}")

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

            # ✅ FIX: Backfill missing bnb-specific keys in older checkpoints
            # CRITICAL: 'alpha' must be set to the learning rate, NOT 0.0 or None!
            for group in self.optimizer.param_groups:
                group.setdefault('alpha', group['lr'])  # Use actual LR
                group.setdefault('t_alpha', None)
                group.setdefault('t_beta3', None)

            self._move_optimizer_state_to_device()

        scheduler_state = checkpoint.get('scheduler_state_dict')
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.start_epoch = checkpoint.get('epoch', 0)
        self.epoch = self.start_epoch
        self.global_step = checkpoint.get('global_step', self.global_step)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)

        print(f"[Trainer] Resumed from epoch {self.start_epoch} (global step {self.global_step})")

    def _move_optimizer_state_to_device(self):
        """Ensure optimizer state tensors live on the right device after loading."""
        if not torch.cuda.is_available():
            return
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def train(self):
        """Main training loop."""
        print("=" * 70)
        print("Stage 1: CLIP-style Vision-Text Pretraining")
        print("=" * 70)
        print(f"Training for {self.config['training']['epochs']} epochs")
        print(f"Effective batch size: {self.config['training']['batch_size'] * self.config['training']['grad_accum']}")
        print("=" * 70)

        if self.start_epoch >= self.config['training']['epochs']:
            print("[Trainer] Checkpoint already completed the requested epochs. Nothing to train.")
            return

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.epoch = epoch + 1

            # Train
            train_loss = self.train_epoch()

            #torch.cuda.empty_cache()

            # Validate
            val_loss, r1_i2t, r1_t2i = self.validate()

            # Print summary
            print(f"\nEpoch {self.epoch}/{self.config['training']['epochs']} Summary:")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Val Loss:       {val_loss:.4f}")
            print(f"  R@1 I→T:        {r1_i2t:.4f}")
            print(f"  R@1 T→I:        {r1_t2i:.4f}")
            print(f"  Current LR:     {self.scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            if self.epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f"checkpoint_epoch{self.epoch}.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                print("  ✨ New best model!")

        # Save final checkpoint
        self.save_checkpoint("final.pt")

        # Finish wandb
        if self.config['wandb']['enabled']:
            wandb.finish()

        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.config['training']['checkpoint_dir']}")
        print("=" * 70)


def main():
    
    # torch.backends.cudnn.benchmark = False  # Tắt auto-tuning
    # torch.backends.cudnn.deterministic = True
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 1 CLIP Training")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Optional checkpoint path to resume from (overrides config)')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config.setdefault('training', {})
        config['training']['resume_from'] = args.resume

    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Create trainer and train
    trainer = Trainer(config)
    print("Starting training...")
    trainer.train()


if __name__ == '__main__':
    main()

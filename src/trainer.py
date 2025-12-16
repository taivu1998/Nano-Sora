"""Training logic for Nano-Sora with Rectified Flow and EMA."""

import torch
import os
from tqdm import tqdm
from .utils import patchify, unpatchify


class EMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of weights that are updated with:
        shadow = decay * shadow + (1 - decay) * current_weights

    This produces smoother weights that typically generate better samples.
    """

    def __init__(self, model, decay: float = 0.9999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.9999 is typical, higher = smoother)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        """Apply shadow weights to model (for inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class Trainer:
    def __init__(self, model, train_dataloader, config, device, logger,
                 val_dataloader=None, use_ema=True, ema_decay=0.9999,
                 start_epoch=1, best_loss=float('inf')):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.logger = logger
        self.start_epoch = start_epoch

        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        if use_ema:
            self.logger.info(f"EMA enabled with decay={ema_decay}")

        # Determine if we can use AMP (only on CUDA)
        self.use_amp = config['training']['use_amp'] and device.type == 'cuda'
        if config['training']['use_amp'] and not self.use_amp:
            self.logger.warning("AMP requested but CUDA not available. Disabling AMP.")

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )

        # Learning rate scheduler (cosine annealing with warmup)
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        total_epochs = config['training']['epochs']

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Avoid division by zero on first epoch
                return max(epoch / warmup_epochs, 0.1)
            else:
                import math
                progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision scaler (only for CUDA)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = None

        # Save directory
        self.save_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['name'])
        os.makedirs(self.save_dir, exist_ok=True)

        # Best model tracking
        self.best_loss = best_loss

    def flow_matching_loss(self, x1):
        """
        Rectified Flow Loss: || v_pred - (x1 - x0) ||^2
        Learns straight-line trajectories from Noise (x0) to Data (x1).

        Key insight: The model predicts velocity in PATCH SPACE, so we need to
        patchify the target velocity (x1 - x0) for comparison.
        """
        b = x1.shape[0]
        x0 = torch.randn_like(x1)  # Noise
        t = torch.rand(b, device=self.device)

        # 1. Linear Interpolation: xt = t * x1 + (1-t) * x0
        t_broad = t.view(b, 1, 1, 1, 1)
        xt = t_broad * x1 + (1 - t_broad) * x0

        # 2. Predict Velocity (model outputs in patch space)
        if self.use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                v_pred = self.model(xt, t)  # (B, N, patch_vol)
                v_target = patchify(x1 - x0, self.model.patch_size)  # (B, N, patch_vol)
                loss = torch.mean((v_pred - v_target) ** 2)
        else:
            v_pred = self.model(xt, t)
            v_target = patchify(x1 - x0, self.model.patch_size)
            loss = torch.mean((v_pred - v_target) ** 2)

        return loss

    @torch.no_grad()
    def validate(self, use_ema=True):
        """Compute validation loss on held-out data."""
        if self.val_dataloader is None:
            return None

        self.model.eval()

        # Apply EMA weights for validation
        if self.use_ema and use_ema:
            self.ema.apply_shadow(self.model)

        total_loss = 0
        num_batches = 0

        for x in self.val_dataloader:
            x = x.to(self.device)

            # For validation, use fixed t values for more stable metrics
            b = x.shape[0]
            x0 = torch.randn_like(x)
            t = torch.rand(b, device=self.device)

            t_broad = t.view(b, 1, 1, 1, 1)
            xt = t_broad * x + (1 - t_broad) * x0

            if self.use_amp:
                with torch.amp.autocast('cuda', enabled=True):
                    v_pred = self.model(xt, t)
                    v_target = patchify(x - x0, self.model.patch_size)
                    loss = torch.mean((v_pred - v_target) ** 2)
            else:
                v_pred = self.model(xt, t)
                v_target = patchify(x - x0, self.model.patch_size)
                loss = torch.mean((v_pred - v_target) ** 2)

            total_loss += loss.item()
            num_batches += 1

        # Restore original weights
        if self.use_ema and use_ema:
            self.ema.restore(self.model)

        return total_loss / num_batches if num_batches > 0 else None

    def train(self):
        self.logger.info("Starting Training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"AMP enabled: {self.use_amp}")
        self.logger.info(f"EMA enabled: {self.use_ema}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        epochs = self.config['training']['epochs']
        validate_every = self.config['training'].get('validate_every', 5)

        if self.start_epoch > 1:
            self.logger.info(f"Resuming from epoch {self.start_epoch}, best_loss={self.best_loss:.6f}")

        for epoch in range(self.start_epoch, epochs + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{epochs}")
            total_loss = 0
            num_batches = 0

            for x in pbar:
                x = x.to(self.device)
                self.optimizer.zero_grad()

                loss = self.flow_matching_loss(x)

                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                    self.optimizer.step()

                # Update EMA after each step
                if self.use_ema:
                    self.ema.update(self.model)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})

            # Step the scheduler
            self.scheduler.step()

            avg_train_loss = total_loss / num_batches

            # Validation
            val_loss = None
            if self.val_dataloader is not None and epoch % validate_every == 0:
                val_loss = self.validate(use_ema=self.use_ema)
                ema_note = " (EMA)" if self.use_ema else ""
                self.logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}{ema_note}")
            else:
                self.logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")

            # Determine if this is the best model
            current_loss = val_loss if val_loss is not None else avg_train_loss
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss

            # Save Checkpoint
            if epoch % self.config['experiment']['save_every'] == 0:
                self.save_checkpoint(epoch, avg_train_loss, val_loss, is_best)
                # Also save as 'latest' for easy resume
                self.save_checkpoint("latest", avg_train_loss, val_loss, is_best=False, epoch_num=epoch)

        # Save final checkpoint
        val_loss = self.validate() if self.val_dataloader else None
        self.save_checkpoint("final", avg_train_loss, val_loss, is_best=False, epoch_num=epochs)
        self.save_checkpoint("latest", avg_train_loss, val_loss, is_best=False, epoch_num=epochs)
        self.logger.info(f"Training complete! Best loss: {self.best_loss:.6f}")

    def save_checkpoint(self, tag, train_loss, val_loss=None, is_best=False, epoch_num=None):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'epoch': epoch_num if epoch_num is not None else tag,
        }

        # Add EMA weights if enabled
        if self.use_ema:
            checkpoint['ema'] = self.ema.state_dict()
            # Also save model with EMA weights applied
            self.ema.apply_shadow(self.model)
            checkpoint['model_ema'] = self.model.state_dict()
            self.ema.restore(self.model)

        path = os.path.join(self.save_dir, f"ckpt_{tag}.pt")
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

        # Save best model separately
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved! Loss: {self.best_loss:.6f}")

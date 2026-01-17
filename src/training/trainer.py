"""
Training Loop for DETR
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import os
import signal
import sys
from pathlib import Path
import time
import gc
from src.training.evaluator import Evaluator

# Mixed precision training
from torch.amp import autocast, GradScaler

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class Trainer:
    """DETR Trainer"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict, device: torch.device,
                 writer: Optional[SummaryWriter] = None):
        """
        Initialize trainer
        
        Args:
            model: DETR model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            writer: TensorBoard writer (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.writer = writer
        
        # Setup mixed precision training (AMP)
        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")
        
        # Setup gradient accumulation
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            print(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
        
        # Memory cleanup frequency
        self.memory_cleanup_frequency = config['training'].get('memory_cleanup_frequency', 10)
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Setup learning rate scheduler
        warmup_epochs = config['lr_schedule']['warmup_epochs']
        num_epochs = config['training']['num_epochs']
        
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs * len(train_loader)
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=(num_epochs - warmup_epochs) * len(train_loader),
                eta_min=config['lr_schedule'].get('min_lr', 1e-6)
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs * len(train_loader)]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs * len(train_loader),
                eta_min=config['lr_schedule'].get('min_lr', 1e-6)
            )
        
        # Setup evaluator
        self.evaluator = Evaluator(config['evaluation'])
        
        # Training state
        self.best_map = 0.0
        self.global_step = 0
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n\nReceived signal {signum}. Saving checkpoint before exit...")
            self.interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        memory_info = {}
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info['ram_gb'] = process.memory_info().rss / (1024 ** 3)
        if torch.cuda.is_available():
            memory_info['gpu_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        return memory_info
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        print_freq = self.config['logging']['print_frequency']
        
        # Initialize gradient accumulation
        accumulation_loss = 0.0
        accumulation_count = 0
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            # DETR expects list of images, not batched tensor
            if isinstance(images, torch.Tensor):
                images = [img.to(self.device) for img in images]
            else:
                images = [img.to(self.device) for img in images]
            
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Convert images to channels-last if enabled
            if hasattr(self.model, 'memory_format') and self.model.memory_format == torch.channels_last:
                images = [img.to(memory_format=torch.channels_last) if isinstance(img, torch.Tensor) else img 
                         for img in images]
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Scale loss by accumulation steps
                scaled_loss = losses / self.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(scaled_loss).backward()
            else:
                # Standard precision
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Scale loss by accumulation steps
                scaled_loss = losses / self.gradient_accumulation_steps
                
                # Backward pass
                scaled_loss.backward()
            
            # Accumulate loss
            accumulation_loss += losses.item()
            accumulation_count += 1
            
            # Only step optimizer after accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients after step
                self.optimizer.zero_grad()
            
            # Scheduler step (every batch, not every accumulation step)
            self.scheduler.step()
            
            # Memory cleanup
            del images, targets, loss_dict, losses
            if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Logging (only on accumulation boundaries or last batch)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_loss = accumulation_loss / accumulation_count if accumulation_count > 0 else 0.0
                total_loss += avg_loss * accumulation_count
                num_batches += accumulation_count
                self.global_step += 1
                
                # Memory monitoring
                memory_info = {}
                if batch_idx % (print_freq * self.gradient_accumulation_steps) == 0:
                    memory_info = self._get_memory_usage()
                    if memory_info:
                        mem_str = ", ".join([f"{k}: {v:.2f}GB" for k, v in memory_info.items()])
                        print(f"Memory: {mem_str}")
                
                # Reduced logging frequency for less I/O overhead
                log_every_n_steps = self.config['logging'].get('log_every_n_steps', 10)
                
                if batch_idx % (print_freq * self.gradient_accumulation_steps) == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
                
                # Less frequent TensorBoard logging
                if self.writer and self.global_step % log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Train/Loss', avg_loss, self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    if memory_info:
                        for key, value in memory_info.items():
                            self.writer.add_scalar(f'Memory/{key}', value, self.global_step)
                
                # Reset accumulation
                accumulation_loss = 0.0
                accumulation_count = 0
        
        # Handle remaining gradients if last batch didn't complete accumulation cycle
        if accumulation_count > 0:
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step with remaining accumulated gradients
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Log final accumulation
            avg_loss = accumulation_loss / accumulation_count
            total_loss += avg_loss * accumulation_count
            num_batches += accumulation_count
            self.global_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # DETR expects list of images
                if isinstance(images, torch.Tensor):
                    images = [img.to(self.device) for img in images]
                else:
                    images = [img.to(self.device) for img in images]
                
                # Get predictions (in eval mode, model returns predictions)
                # Use autocast for validation too if AMP is enabled
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Store for evaluation
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    all_predictions.append(output)
                    all_targets.append(target)
                
                # Memory cleanup during validation
                del images, targets, outputs
                if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Evaluate
        map_score = self.evaluator.evaluate(all_predictions, all_targets)
        
        print(f"Validation mAP: {map_score:.4f}")
        
        if self.writer:
            self.writer.add_scalar('Val/mAP', map_score, epoch)
        
        # Final cleanup after validation
        del all_predictions, all_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return map_score
    
    def save_checkpoint(self, epoch: int, map_score: float, is_best: bool = False, 
                       is_interrupt: bool = False, lightweight: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            map_score: Validation mAP score (0.0 if not validated)
            is_best: Whether this is the best model so far
            is_interrupt: Whether this is an interrupt/error save
            lightweight: If True, save only model weights (faster, for frequent saves)
        """
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if lightweight:
            # Lightweight checkpoint: just model weights and epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_lightweight.pth"
        else:
            # Full checkpoint: includes optimizer, scheduler, metrics
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'map': map_score,
                'config': self.config
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if is_interrupt:
            print(f"Saved interrupt checkpoint: {checkpoint_path}")
        elif lightweight:
            print(f"Saved lightweight checkpoint: {checkpoint_path}")
        else:
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model (only for full checkpoints with validation)
        if is_best and not lightweight:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with mAP: {map_score:.4f}")
        
        # Save latest checkpoint (always overwrite)
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def train(self, start_epoch: int = 0, num_epochs: int = 50):
        """Main training loop"""
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        
        try:
            for epoch in range(start_epoch, num_epochs):
                if self.interrupted:
                    print("\nTraining interrupted. Saving checkpoint...")
                    # Save checkpoint without validation
                    self.save_checkpoint(epoch - 1, 0.0, is_best=False, is_interrupt=True)
                    break
                
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*50}")
                
                # Train
                train_loss = self.train_epoch(epoch)
                print(f"Training Loss: {train_loss:.4f}")
                
                # Save checkpoint every epoch (for safety, even if not validating)
                # This ensures we don't lose progress if training stops early
                save_every_epoch = self.config['checkpoint'].get('save_every_epoch', True)
                if save_every_epoch:
                    # Save lightweight checkpoint every epoch
                    self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=False, lightweight=True)
                
                # Validate less frequently for speed (every 10 epochs instead of 5)
                validate_frequency = 10
                if (epoch + 1) % validate_frequency == 0 or epoch == num_epochs - 1:
                    map_score = self.validate(epoch)
                    
                    # Save full checkpoint with validation metrics
                    is_best = map_score > self.best_map
                    if is_best:
                        self.best_map = map_score
                    
                    if (epoch + 1) % self.config['checkpoint']['save_frequency'] == 0:
                        self.save_checkpoint(epoch, map_score, is_best, is_interrupt=False, lightweight=False)
                
                print(f"{'='*50}\n")
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=True)
            raise
        
        except Exception as e:
            print(f"\n\nTraining error: {e}")
            print("Saving checkpoint before exit...")
            self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=True)
            raise

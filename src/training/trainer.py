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
from pathlib import Path
import time
from src.training.evaluator import Evaluator


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
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        print_freq = self.config['logging']['print_frequency']
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            # DETR expects list of images, not batched tensor
            if isinstance(images, torch.Tensor):
                images = [img.to(self.device) for img in images]
            else:
                images = [img.to(self.device) for img in images]
            
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += losses.item()
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {losses.item():.4f} LR: {current_lr:.6f}")
                
                if self.writer:
                    self.writer.add_scalar('Train/Loss', losses.item(), self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # DETR expects list of images
                if isinstance(images, torch.Tensor):
                    images = [img.to(self.device) for img in images]
                else:
                    images = [img.to(self.device) for img in images]
                
                # Get predictions (in eval mode, model returns predictions)
                outputs = self.model(images)
                
                # Store for evaluation
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    all_predictions.append(output)
                    all_targets.append(target)
        
        # Evaluate
        map_score = self.evaluator.evaluate(all_predictions, all_targets)
        
        print(f"Validation mAP: {map_score:.4f}")
        
        if self.writer:
            self.writer.add_scalar('Val/mAP', map_score, epoch)
        
        return map_score
    
    def save_checkpoint(self, epoch: int, map_score: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'map': map_score,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with mAP: {map_score:.4f}")
    
    def train(self, start_epoch: int = 0, num_epochs: int = 50):
        """Main training loop"""
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                map_score = self.validate(epoch)
                
                # Save checkpoint
                is_best = map_score > self.best_map
                if is_best:
                    self.best_map = map_score
                
                if (epoch + 1) % self.config['checkpoint']['save_frequency'] == 0:
                    self.save_checkpoint(epoch, map_score, is_best)
            
            print(f"{'='*50}\n")

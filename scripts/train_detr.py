#!/usr/bin/env python3
"""
Main training script for RF-DETR model
"""
import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset import CocoDataset
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.collate import collate_fn
from src.training.model import get_detr_model
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device() -> torch.device:
    """Setup GPU/CPU device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR model locally")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="datasets/train",
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="datasets/val",
        help="Path to validation dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = setup_device()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Setup TensorBoard
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CocoDataset(
        dataset_dir=args.train_dir,
        transforms=get_train_transforms(config['augmentation']['train'])
    )
    
    val_dataset = CocoDataset(
        dataset_dir=args.val_dir,
        transforms=get_val_transforms(config['augmentation']['val'])
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        collate_fn=collate_fn
    )
    
    # Create model
    print("Initializing model...")
    model = get_detr_model(config['model'])
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        writer=writer
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Train
    print("Starting training...")
    trainer.train(start_epoch=start_epoch, num_epochs=config['training']['num_epochs'])
    
    print("Training complete!")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train DETR model on a single frame for 1 epoch
Used for active learning: fine-tune on manually labeled frame 0
"""
import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset import CocoDataset
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.collate import collate_fn
from src.training.model import get_detr_model
from src.training.trainer import Trainer


def setup_device() -> torch.device:
    """Setup GPU/CPU device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_single_frame(
    dataset_dir: str,
    output_checkpoint_path: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    batch_size: int = 1
):
    """
    Train model on single frame dataset
    
    Args:
        dataset_dir: Directory containing images/ and annotations/annotations.json
        output_checkpoint_path: Path to save trained checkpoint
        num_epochs: Number of epochs (default: 1)
        learning_rate: Learning rate (default: 1e-5)
        batch_size: Batch size (default: 1)
    """
    device = setup_device()
    
    # Create dataset with training transforms
    # Use minimal augmentation config for single frame training
    aug_config = {
        'horizontal_flip': 0,  # No flipping for single frame
        'color_jitter': {
            'brightness': 0.0,
            'contrast': 0.0,
            'saturation': 0.0,
            'hue': 0.0
        },
        'clahe': {'enabled': False},
        'normalize': True
    }
    train_transforms = get_train_transforms(aug_config)
    train_dataset = CocoDataset(dataset_dir, transforms=train_transforms)
    
    print(f"Dataset size: {len(train_dataset)} samples")
    
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty! Check dataset directory structure.")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Single frame, no need for multiple workers
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create empty validation loader (not needed for single frame)
    val_transforms = get_val_transforms(aug_config)
    val_dataset = CocoDataset(dataset_dir, transforms=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model config (pretrained DETR)
    model_config = {
        'num_classes': 2,  # player, ball
        'pretrained': True  # Start from pretrained weights
    }
    
    training_config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs
    }
    
    # Create model
    print("\nCreating model (pretrained DETR)...")
    model = get_detr_model(model_config, training_config)
    model = model.to(device)
    
    # Create trainer config with all required keys
    trainer_config = {
        'model': model_config,
        'training': training_config,
        'optimizer': {
            'lr': learning_rate,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        },
        'lr_schedule': {
            'warmup_epochs': 0,  # No warmup for single-frame training
            'min_lr': 1e-6
        },
        'evaluation': {
            'compute_map': False  # Skip mAP computation for single-frame training
        },
        'logging': {
            'print_frequency': 10
        }
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        writer=None,  # No TensorBoard for single-frame training
        mlflow_run=None  # No MLflow for single-frame training
    )
    
    # Train for specified epochs
    print(f"\nTraining for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        trainer.train_epoch(epoch)
    
    # Save checkpoint
    print(f"\nSaving checkpoint to: {output_checkpoint_path}")
    output_path = Path(output_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'model': model_config,
            'training': training_config
        },
        'epoch': num_epochs
    }
    
    torch.save(checkpoint, output_checkpoint_path)
    print(f"Checkpoint saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train DETR on single frame")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing images/ and annotations/annotations.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save trained checkpoint"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    
    args = parser.parse_args()
    
    train_single_frame(
        dataset_dir=args.dataset_dir,
        output_checkpoint_path=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

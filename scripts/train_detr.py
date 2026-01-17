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

# MLflow tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

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
    """Setup GPU/CPU device with optimizations"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Faster, but non-deterministic
        
        # Enable TF32 on Ampere GPUs (A40) for faster matmul operations
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for Ampere GPU (faster matmul operations)")
        
        print("CUDA optimizations enabled")
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
    
    # Setup MLflow (after datasets are created)
    mlflow_run = None
    if config['logging'].get('mlflow', False) and MLFLOW_AVAILABLE:
        try:
            # Set tracking URI
            tracking_uri = config['logging'].get('mlflow_tracking_uri', 'file:./mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set experiment
            experiment_name = config['logging'].get('mlflow_experiment_name', 'detr_training')
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except Exception:
                # Experiment already exists
                experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
            # Start run
            mlflow_run = mlflow.start_run(experiment_id=experiment_id)
            
            # Log hyperparameters
            mlflow.log_params({
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'num_epochs': config['training']['num_epochs'],
                'weight_decay': config['training']['weight_decay'],
                'gradient_clip': config['training'].get('gradient_clip', 0),
                'gradient_accumulation_steps': config['training'].get('gradient_accumulation_steps', 1),
                'mixed_precision': config['training'].get('mixed_precision', False),
                'compile_model': config['training'].get('compile_model', False),
                'channels_last': config['training'].get('channels_last', False),
                'num_workers': config['dataset']['num_workers'],
                'prefetch_factor': config['dataset'].get('prefetch_factor', 2),
            })
            
            # Log model architecture parameters
            mlflow.log_params({
                'model_architecture': config['model'].get('architecture', 'detr'),
                'backbone': config['model'].get('backbone', 'resnet50'),
                'num_classes': config['model'].get('num_classes', 2),
                'hidden_dim': config['model'].get('hidden_dim', 256),
                'num_encoder_layers': config['model'].get('num_encoder_layers', 6),
                'num_decoder_layers': config['model'].get('num_decoder_layers', 6),
            })
            
            # Log dataset info
            mlflow.log_params({
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
            })
            
            print(f"MLflow tracking enabled. Run ID: {mlflow_run.info.run_id}")
            print(f"MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize MLflow: {e}")
            mlflow_run = None
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['dataset'].get('prefetch_factor', 2),
        persistent_workers=config['dataset'].get('persistent_workers', False) if config['dataset']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['dataset'].get('prefetch_factor', 2),
        persistent_workers=config['dataset'].get('persistent_workers', False) if config['dataset']['num_workers'] > 0 else False
    )
    
    # Create model
    print("Initializing model...")
    model = get_detr_model(config['model'], training_config=config.get('training', {}))
    model = model.to(device)
    
    # Convert to channels-last memory format for faster convolutions
    if config['training'].get('channels_last', False) and device.type == 'cuda':
        try:
            print("Converting model to channels-last memory format...")
            model = model.to(memory_format=torch.channels_last)
            print("Model converted to channels-last format")
        except Exception as e:
            print(f"Warning: Channels-last conversion failed: {e}")
            print("Continuing with default memory format...")
    
    # Compile model for optimization (PyTorch 2.0+)
    if config['training'].get('compile_model', False):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")
            print("Continuing without compilation...")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        writer=writer,
        mlflow_run=mlflow_run
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Handle torch.compile prefix mismatch
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Checkpoint was saved with torch.compile, but we're loading without it
            print("Removing torch.compile prefixes from checkpoint...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_key = k.replace('_orig_mod.', '', 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # Load with strict=False to handle any remaining mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        
        start_epoch = checkpoint['epoch'] + 1
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Train
    print("Starting training...")
    trainer.train(start_epoch=start_epoch, num_epochs=config['training']['num_epochs'])
    
    print("Training complete!")
    if writer:
        writer.close()
    
    # End MLflow run
    if mlflow_run:
        try:
            mlflow.end_run()
            print("MLflow run completed")
        except Exception as e:
            print(f"Warning: Failed to end MLflow run: {e}")


if __name__ == "__main__":
    main()

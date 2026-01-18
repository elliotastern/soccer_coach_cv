#!/usr/bin/env python3
"""
Export trained DETR model for inference
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.training.model import get_detr_model
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_model(checkpoint_path: str, output_path: str, config_path: str = "configs/training.yaml"):
    """
    Export trained model for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save exported model
        config_path: Path to training config
    """
    # Load config
    config = load_config(config_path)
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = get_detr_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save model
    print(f"Exporting model to: {output_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'map': checkpoint.get('map', 0.0),
        'epoch': checkpoint.get('epoch', 0)
    }, output_path)
    
    print("Model exported successfully!")


def main():
    parser = argparse.ArgumentParser(description="Export trained DETR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config"
    )
    
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()

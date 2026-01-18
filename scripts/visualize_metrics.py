#!/usr/bin/env python3
"""
Visualize training metrics over epochs from TensorBoard logs and checkpoints
"""
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_tensorboard_metrics(log_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract metrics from TensorBoard event files
    
    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    if not TENSORBOARD_AVAILABLE:
        return {}
    
    metrics = {}
    
    try:
        # Find all event files
        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        if not event_files:
            print(f"No TensorBoard event files found in {log_dir}")
            return {}
        
        # Use the most recent event file
        latest_event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        print(f"Reading TensorBoard events from: {latest_event_file}")
        
        # Create EventAccumulator
        ea = EventAccumulator(str(latest_event_file.parent))
        ea.Reload()
        
        # Get all scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            metrics[tag] = [(event.step, event.value) for event in scalar_events]
        
        print(f"Extracted {len(metrics)} metrics from TensorBoard")
        
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        return {}
    
    return metrics


def extract_checkpoint_metrics(checkpoint_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract metrics from checkpoint files
    
    Returns:
        Dictionary mapping metric names to list of (epoch, value) tuples
    """
    if not TORCH_AVAILABLE:
        return {}
    
    metrics = {'epoch': [], 'mAP': [], 'loss': []}
    
    try:
        checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pth"))
        checkpoint_files.sort(key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else -1)
        
        for ckpt_path in checkpoint_files:
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', -1)
                if epoch >= 0:
                    metrics['epoch'].append(epoch)
                    metrics['mAP'].append(checkpoint.get('map', 0.0))
                    # Try to get loss from checkpoint if available
                    if 'loss' in checkpoint:
                        metrics['loss'].append(checkpoint['loss'])
            except Exception as e:
                print(f"Warning: Could not load {ckpt_path}: {e}")
                continue
        
        print(f"Extracted metrics from {len(metrics['epoch'])} checkpoints")
        
    except Exception as e:
        print(f"Error reading checkpoints: {e}")
    
    return metrics


def plot_metrics(tensorboard_metrics: Dict, checkpoint_metrics: Dict, output_path: str):
    """
    Create visualization plots for training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    if 'Train/Loss' in tensorboard_metrics:
        steps, losses = zip(*tensorboard_metrics['Train/Loss'])
        ax1.plot(steps, losses, 'b-', linewidth=2, label='Training Loss', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No training loss data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training Loss')
    
    # 2. Validation mAP
    ax2 = axes[0, 1]
    if 'Val/mAP' in tensorboard_metrics:
        steps, maps = zip(*tensorboard_metrics['Val/mAP'])
        ax2.plot(steps, maps, 'g-', linewidth=2, marker='o', markersize=6, label='Validation mAP', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    elif 'mAP' in checkpoint_metrics and checkpoint_metrics['mAP']:
        epochs = checkpoint_metrics['epoch']
        maps = checkpoint_metrics['mAP']
        ax2.plot(epochs, maps, 'g-', linewidth=2, marker='o', markersize=6, label='Validation mAP', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No validation mAP data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Validation mAP')
    
    # 3. Learning Rate
    ax3 = axes[1, 0]
    if 'Train/LearningRate' in tensorboard_metrics:
        steps, lrs = zip(*tensorboard_metrics['Train/LearningRate'])
        ax3.plot(steps, lrs, 'r-', linewidth=2, label='Learning Rate', alpha=0.7)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule')
    
    # 4. Memory Usage
    ax4 = axes[1, 1]
    memory_metrics = {k: v for k, v in tensorboard_metrics.items() if k.startswith('Memory/')}
    if memory_metrics:
        for metric_name, values in memory_metrics.items():
            steps, mem_values = zip(*values)
            label = metric_name.replace('Memory/', '').replace('_', ' ').title()
            ax4.plot(steps, mem_values, linewidth=2, label=label, alpha=0.7)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Memory (GB)')
        ax4.set_title('Memory Usage')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No memory usage data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Memory Usage')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Also create a detailed epoch-based plot if we have checkpoint data
    if checkpoint_metrics and checkpoint_metrics.get('mAP'):
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = checkpoint_metrics['epoch']
        maps = checkpoint_metrics['mAP']
        ax.plot(epochs, maps, 'g-o', linewidth=2, markersize=8, label='Validation mAP')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('Validation mAP Over Epochs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add value annotations
        for i, (epoch, map_val) in enumerate(zip(epochs, maps)):
            if i % max(1, len(epochs) // 10) == 0 or i == len(epochs) - 1:  # Annotate every 10th or last
                ax.annotate(f'{map_val:.3f}', (epoch, map_val), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        epoch_plot_path = output_path.replace('.png', '_epochs.png')
        plt.savefig(epoch_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved epoch-based plot to: {epoch_plot_path}")
        plt.close(fig2)
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory containing TensorBoard event files"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_metrics.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    # Extract metrics
    print("Extracting metrics from TensorBoard logs...")
    tensorboard_metrics = extract_tensorboard_metrics(args.log_dir)
    
    print("Extracting metrics from checkpoints...")
    checkpoint_metrics = extract_checkpoint_metrics(args.checkpoint_dir)
    
    # Create visualization
    print("Creating visualization...")
    plot_metrics(tensorboard_metrics, checkpoint_metrics, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()

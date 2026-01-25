#!/usr/bin/env python3
"""
Monitor RF-DETR training and generate predictions after each epoch
Watches for new checkpoints and runs inference on fixed frames
"""
import time
import json
from pathlib import Path
import subprocess
import argparse
from datetime import datetime


def get_latest_epoch_from_log(log_file):
    """Get the latest epoch number from log file"""
    if not log_file.exists():
        return None
    
    latest_epoch = None
    with open(log_file) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    epoch = data.get('epoch')
                    if epoch is not None:
                        latest_epoch = max(latest_epoch, epoch) if latest_epoch is not None else epoch
                except:
                    pass
    
    return latest_epoch


def get_latest_epoch_from_checkpoint(checkpoint_path):
    """Get epoch number from checkpoint"""
    if not checkpoint_path.exists():
        return None
    
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return ckpt.get('epoch')
    except:
        return None


def monitor_and_generate_predictions(
    dataset_dir,
    checkpoint_dir,
    output_dir,
    log_file,
    threshold=0.3,
    check_interval=60
):
    """Monitor training and generate predictions after each epoch"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pth"
    processed_epochs = set()
    
    print("=" * 70)
    print("TRAINING MONITOR - Epoch Predictions Generator")
    print("=" * 70)
    print(f"Dataset: {dataset_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Check interval: {check_interval} seconds")
    print()
    print("Monitoring for new epochs...")
    print()
    
    while True:
        try:
            # Check log file for new epochs
            latest_log_epoch = get_latest_epoch_from_log(Path(log_file))
            
            # Check checkpoint for current epoch
            checkpoint_epoch = get_latest_epoch_from_checkpoint(checkpoint_path)
            
            # Use checkpoint epoch as it's more reliable
            current_epoch = checkpoint_epoch if checkpoint_epoch is not None else latest_log_epoch
            
            if current_epoch is not None and current_epoch not in processed_epochs:
                print(f"📊 New epoch detected: {current_epoch}")
                print(f"   Generating predictions...")
                
                # Run prediction script
                cmd = [
                    "python3", "scripts/epoch_predictions_monitor.py",
                    "--dataset-dir", str(dataset_dir),
                    "--checkpoint-path", str(checkpoint_path),
                    "--output-dir", str(output_dir),
                    "--epoch", str(current_epoch),
                    "--threshold", str(threshold)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/soccer_coach_cv")
                
                if result.returncode == 0:
                    print(f"   ✅ Epoch {current_epoch} predictions generated")
                    processed_epochs.add(current_epoch)
                    
                    # Print HTML viewer location
                    html_path = output_dir / "epoch_predictions_viewer.html"
                    if html_path.exists():
                        print(f"   📄 Viewer: {html_path}")
                else:
                    print(f"   ⚠️  Error generating predictions:")
                    print(result.stderr)
            
            # Sleep before next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(check_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor training and generate epoch predictions")
    parser.add_argument("--dataset-dir", type=str, default="datasets/rf_detr_soccertrack", help="Dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="models/rf_detr_soccertrack", help="Checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="models/rf_detr_soccertrack/epoch_predictions", help="Output directory")
    parser.add_argument("--log-file", type=str, default="models/rf_detr_soccertrack/log.txt", help="Training log file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    monitor_and_generate_predictions(
        args.dataset_dir,
        args.checkpoint_dir,
        args.output_dir,
        args.log_file,
        args.threshold,
        args.check_interval
    )

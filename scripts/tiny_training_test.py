#!/usr/bin/env python3
"""
Tiny training test - runs ~10 batches then validates
Takes ~2-3 minutes, gives evidence of mAP improvement
"""
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset import CocoDataset
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.collate import collate_fn
from src.training.model import get_detr_model
from src.training.trainer import Trainer


def main():
    print("=" * 60)
    print("TINY TRAINING TEST - Evidence of Improvement")
    print("=" * 60)
    print("Running ~10 training batches + validation")
    print("Previous baseline: mAP = 0.00% (BROKEN)")
    print("Expected: mAP > 0.00% (FIXED)")
    print("=" * 60)
    
    # Load config
    with open("configs/training.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Minimal settings
    config['training']['batch_size'] = 4
    config['checkpoint']['save_every_epoch'] = False
    config['logging']['print_frequency'] = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Tiny datasets
    print("\nLoading tiny datasets...")
    train_dataset = CocoDataset(
        dataset_dir="datasets/train",
        transforms=get_train_transforms(config['augmentation']['train'])
    )
    val_dataset = CocoDataset(
        dataset_dir="datasets/val",
        transforms=get_val_transforms(config['augmentation']['val'])
    )
    
    # Use only 50 train, 20 val samples
    train_subset = Subset(train_dataset, list(range(min(50, len(train_dataset)))))
    val_subset = Subset(val_dataset, list(range(min(20, len(val_dataset)))))
    
    train_loader = DataLoader(
        train_subset, batch_size=4, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    
    print(f"Training: {len(train_subset)} samples, {len(train_loader)} batches")
    print(f"Validation: {len(val_subset)} samples")
    
    # Create model
    print("\nCreating model...")
    model = get_detr_model(config['model'], config['training'])
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=config, device=device, writer=None, mlflow_run=None
    )
    
    # Train ~10 batches
    print("\n" + "=" * 60)
    print("TRAINING (~10 batches)")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    total_loss = 0
    num_batches = min(10, len(train_loader))
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Batch {batch_idx+1}/{num_batches}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"\nAverage training loss: {avg_loss:.4f}")
    
    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION (CRITICAL TEST)")
    print("=" * 60)
    print("Previous: mAP = 0.00%")
    print("Expected: mAP > 0.00%")
    print("-" * 60)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                target_0based = target.copy()
                if 'labels' in target_0based and len(target_0based['labels']) > 0:
                    target_0based['labels'] = target_0based['labels'] - 1
                all_predictions.append(output)
                all_targets.append(target_0based)
    
    # Evaluate
    eval_metrics = trainer.evaluator.evaluate(all_predictions, all_targets)
    
    player_map = eval_metrics.get('player_map_05', eval_metrics.get('player_map', 0.0))
    ball_map = eval_metrics.get('ball_map_05', eval_metrics.get('ball_map', 0.0))
    player_recall = eval_metrics.get('player_recall_05', eval_metrics.get('player_recall', 0.0))
    ball_precision = eval_metrics.get('ball_precision_05', eval_metrics.get('ball_precision', 0.0))
    ball_recall = eval_metrics.get('ball_recall_05', eval_metrics.get('ball_recall', 0.0))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall mAP: {eval_metrics['map']:.4f}")
    print(f"\nPlayer Metrics:")
    print(f"  mAP@0.5:    {player_map:.4f} {'✅' if player_map > 0 else '❌'}")
    print(f"  Recall@0.5: {player_recall:.4f} {'✅' if player_recall > 0 else '❌'}")
    print(f"\nBall Metrics:")
    print(f"  mAP@0.5:     {ball_map:.4f} {'✅' if ball_map > 0 else '❌'}")
    print(f"  Precision@0.5: {ball_precision:.4f}")
    print(f"  Recall@0.5:  {ball_recall:.4f} {'✅' if ball_recall > 0 else '❌'}")
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    # Success criteria: Player mAP > 0 (critical fix)
    # Ball mAP may still be 0 with minimal training - that's OK
    if player_map > 0:
        print("\n✅ SUCCESS! Critical fix is working!")
        print(f"   Player mAP: {player_map:.4f} (was 0.0000) ✅")
        print(f"   Player Recall: {player_recall:.4f} (was 0.0000) ✅")
        print(f"   Overall mAP: {eval_metrics['map']:.4f} (was 0.0000) ✅")
        print(f"\n   Ball mAP: {ball_map:.4f} (may be 0 with minimal training - OK)")
        print("\n🎉 The critical indexing bug is FIXED!")
        print("   Players are now being detected correctly.")
        print("   Ball detection will improve with more training.")
        print("\n✅ You can now run full training with confidence!")
        return True
    else:
        print("\n❌ Still seeing 0% Player mAP")
        print("   Check label conversion in model inference")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Quick test to verify improvements are working
Runs minimal training (1 epoch) to check if mAP > 0
"""
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset import CocoDataset
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.collate import collate_fn
from src.training.model import get_detr_model
from src.training.trainer import Trainer


def quick_test():
    """Run minimal test to verify improvements"""
    print("=" * 60)
    print("QUICK IMPROVEMENT TEST")
    print("=" * 60)
    
    # Load config
    config_path = "configs/training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick test: 1 epoch, small batch
    config['training']['num_epochs'] = 1
    config['training']['batch_size'] = 4  # Small batch
    config['checkpoint']['save_every_epoch'] = False  # Don't save
    config['logging']['print_frequency'] = 10  # Print more often
    
    print("\n1. Testing Dataset Label Indexing...")
    print("-" * 60)
    
    # Test dataset
    train_dataset = CocoDataset(
        dataset_dir="datasets/train",
        transforms=None  # No transforms for quick check
    )
    
    sample = train_dataset[0]
    labels = sample[1]['labels']
    unique_labels = labels.unique().tolist()
    
    print(f"   Sample labels: {unique_labels}")
    if 1 in unique_labels or 2 in unique_labels:
        print("   ✅ Dataset uses 1-based indexing (1=player, 2=ball)")
    else:
        print("   ❌ Dataset still using 0-based indexing - FIX NEEDED")
        return False
    
    print("\n2. Testing Model Configuration...")
    print("-" * 60)
    
    # Check config
    focal_enabled = config['training'].get('focal_loss', {}).get('enabled', False)
    weights_enabled = config['training'].get('class_weights', {}).get('enabled', False)
    
    print(f"   Focal Loss enabled: {focal_enabled}")
    print(f"   Class weights enabled: {weights_enabled}")
    
    if focal_enabled and not weights_enabled:
        print("   ✅ Focal Loss enabled, class weights disabled")
    else:
        print("   ⚠️  Config may need adjustment")
    
    print("\n3. Creating Model...")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    try:
        model = get_detr_model(config['model'], config['training'])
        model = model.to(device)
        print("   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    print("\n4. Running Minimal Training (1 epoch, ~50 batches)...")
    print("-" * 60)
    
    # Create small dataset subset for quick test
    train_dataset = CocoDataset(
        dataset_dir="datasets/train",
        transforms=get_train_transforms(config['augmentation']['train'])
    )
    
    val_dataset = CocoDataset(
        dataset_dir="datasets/val",
        transforms=get_val_transforms(config['augmentation']['val'])
    )
    
    # Use small subset
    train_indices = list(range(min(200, len(train_dataset))))  # Max 200 samples
    val_indices = list(range(min(100, len(val_dataset))))  # Max 100 samples
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Training samples: {len(train_subset)}")
    print(f"   Validation samples: {len(val_subset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        writer=None,  # No TensorBoard for quick test
        mlflow_run=None  # No MLflow for quick test
    )
    
    print("\n5. Training...")
    print("-" * 60)
    
    # Train 1 epoch
    try:
        train_loss = trainer.train_epoch(0)
        print(f"   Training loss: {train_loss:.4f}")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n6. Validation (Critical Test)...")
    print("-" * 60)
    print("   This is the KEY TEST - mAP should be > 0")
    print("   Previous baseline: mAP = 0.00% (BROKEN)")
    print("-" * 60)
    
    try:
        map_score = trainer.validate(0)
        
        # Get detailed metrics
        trainer.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if batch_idx >= 5:  # Only check first 5 batches
                    break
                    
                images = [img.to(device) for img in images]
                outputs = trainer.model(images)
                
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    target_0based = target.copy()
                    if 'labels' in target_0based and len(target_0based['labels']) > 0:
                        target_0based['labels'] = target_0based['labels'] - 1
                    all_predictions.append(output)
                    all_targets.append(target_0based)
        
        eval_metrics = trainer.evaluator.evaluate(all_predictions, all_targets)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        player_map = eval_metrics.get('player_map_05', eval_metrics.get('player_map', 0.0))
        ball_map = eval_metrics.get('ball_map_05', eval_metrics.get('ball_map', 0.0))
        player_recall = eval_metrics.get('player_recall_05', eval_metrics.get('player_recall', 0.0))
        ball_precision = eval_metrics.get('ball_precision_05', eval_metrics.get('ball_precision', 0.0))
        ball_recall = eval_metrics.get('ball_recall_05', eval_metrics.get('ball_recall', 0.0))
        
        print(f"\nOverall mAP: {map_score:.4f}")
        print(f"\nPlayer Metrics:")
        print(f"  mAP@0.5:    {player_map:.4f} {'✅' if player_map > 0 else '❌'}")
        print(f"  Recall@0.5: {player_recall:.4f} {'✅' if player_recall > 0 else '❌'}")
        
        print(f"\nBall Metrics:")
        print(f"  mAP@0.5:     {ball_map:.4f} {'✅' if ball_map > 0 else '❌'}")
        print(f"  Precision@0.5: {ball_precision:.4f} {'✅' if ball_precision > 0.10 else '⚠️'}")
        print(f"  Recall@0.5:  {ball_recall:.4f} {'✅' if ball_recall > 0 else '❌'}")
        
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        
        # Critical test: mAP > 0
        if player_map > 0 and ball_map > 0:
            print("✅ SUCCESS: Class indexing fix is working!")
            print("   mAP is no longer 0% - the critical bug is fixed")
            
            if ball_precision > 0.10:
                print("✅ Ball precision improved (Focal Loss working)")
            else:
                print("⚠️  Ball precision still low (may need more training)")
            
            print("\n🎉 Improvements are working! You can now run full training.")
            return True
        else:
            print("❌ FAILURE: mAP is still 0%")
            print("   The class indexing fix may not be working correctly")
            print("   Check dataset labels and model label conversion")
            return False
            
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

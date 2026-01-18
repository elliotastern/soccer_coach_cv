#!/usr/bin/env python3
"""
Train to epoch 5 and evaluate progress
"""
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset import CocoDataset
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.collate import collate_fn
from src.training.model import get_detr_model
from src.training.trainer import Trainer


def main():
    print("=" * 70)
    print("TRAINING TO EPOCH 5 - PROGRESS EVALUATION")
    print("=" * 70)
    
    # Load config
    config_path = "configs/training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for 5 epochs
    config['training']['num_epochs'] = 5
    config['checkpoint']['save_every_epoch'] = True
    config['logging']['print_frequency'] = 20
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    print("\n" + "=" * 70)
    print("Loading datasets...")
    print("=" * 70)
    
    train_dataset = CocoDataset(
        dataset_dir="datasets/train",
        transforms=get_train_transforms(config['augmentation']['train'])
    )
    
    val_dataset = CocoDataset(
        dataset_dir="datasets/val",
        transforms=get_val_transforms(config['augmentation']['val'])
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        prefetch_factor=config['dataset'].get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        prefetch_factor=config['dataset'].get('prefetch_factor', 2)
    )
    
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("Creating model...")
    print("=" * 70)
    
    model = get_detr_model(config['model'], config['training'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        writer=None,  # No TensorBoard for this run
        mlflow_run=None  # No MLflow for this run
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING (5 EPOCHS)")
    print("=" * 70)
    print("\nBaseline (before fixes):")
    print("  Player mAP: 0.0000")
    print("  Ball mAP: 0.0000")
    print("  Overall mAP: 0.0000")
    print("\n" + "-" * 70)
    
    for epoch in range(5):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/5")
        print(f"{'='*70}")
        
        # Train
        train_loss = trainer.train_epoch(epoch)
        print(f"\nEpoch {epoch + 1} Training Loss: {train_loss:.4f}")
        
        # Validate every epoch
        print(f"\nValidating epoch {epoch + 1}...")
        map_score = trainer.validate(epoch)
        
        # Get detailed metrics
        trainer.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = [img.to(device) for img in images]
                outputs = trainer.model(images)
                
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    target_0based = target.copy()
                    if 'labels' in target_0based and len(target_0based['labels']) > 0:
                        target_0based['labels'] = target_0based['labels'] - 1
                    all_predictions.append(output)
                    all_targets.append(target_0based)
        
        eval_metrics = trainer.evaluator.evaluate(all_predictions, all_targets)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1} RESULTS")
        print(f"{'='*70}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"\nOverall Metrics:")
        print(f"  mAP:        {eval_metrics['map']:.4f}")
        print(f"  Precision:  {eval_metrics['precision']:.4f}")
        print(f"  Recall:     {eval_metrics['recall']:.4f}")
        print(f"  F1:         {eval_metrics['f1']:.4f}")
        
        print(f"\nPlayer Metrics (IoU 0.5):")
        print(f"  mAP@0.5:    {eval_metrics.get('player_map_05', eval_metrics.get('player_map', 0.0)):.4f}")
        print(f"  Precision:  {eval_metrics.get('player_precision_05', eval_metrics.get('player_precision', 0.0)):.4f}")
        print(f"  Recall:     {eval_metrics.get('player_recall_05', eval_metrics.get('player_recall', 0.0)):.4f}")
        print(f"  F1:         {eval_metrics.get('player_f1', 0.0):.4f}")
        
        print(f"\nBall Metrics (IoU 0.5):")
        print(f"  mAP@0.5:    {eval_metrics.get('ball_map_05', eval_metrics.get('ball_map', 0.0)):.4f}")
        print(f"  Precision:  {eval_metrics.get('ball_precision_05', eval_metrics.get('ball_precision', 0.0)):.4f}")
        print(f"  Recall:     {eval_metrics.get('ball_recall_05', eval_metrics.get('ball_recall', 0.0)):.4f}")
        print(f"  F1:         {eval_metrics.get('ball_f1', 0.0):.4f}")
        
        if 'ball_avg_predictions_per_image' in eval_metrics:
            print(f"  Avg preds per image: {eval_metrics['ball_avg_predictions_per_image']:.2f}")
        
        print(f"{'='*70}\n")
    
    # Final evaluation summary
    print("\n" + "=" * 70)
    print("FINAL PROGRESS EVALUATION (EPOCH 5)")
    print("=" * 70)
    
    # Get final metrics
    trainer.model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            outputs = trainer.model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                target_0based = target.copy()
                if 'labels' in target_0based and len(target_0based['labels']) > 0:
                    target_0based['labels'] = target_0based['labels'] - 1
                all_predictions.append(output)
                all_targets.append(target_0based)
    
    final_metrics = trainer.evaluator.evaluate(all_predictions, all_targets)
    
    print("\n📊 COMPARISON: Before vs After 5 Epochs")
    print("-" * 70)
    print(f"{'Metric':<25} {'Before':<15} {'After 5 Epochs':<15} {'Change':<15}")
    print("-" * 70)
    
    player_map_final = final_metrics.get('player_map_05', final_metrics.get('player_map', 0.0))
    ball_map_final = final_metrics.get('ball_map_05', final_metrics.get('ball_map', 0.0))
    player_recall_final = final_metrics.get('player_recall_05', final_metrics.get('player_recall', 0.0))
    ball_precision_final = final_metrics.get('ball_precision_05', final_metrics.get('ball_precision', 0.0))
    ball_recall_final = final_metrics.get('ball_recall_05', final_metrics.get('ball_recall', 0.0))
    
    print(f"{'Overall mAP':<25} {'0.0000':<15} {final_metrics['map']:.4f}{'':<11} ✅ +{final_metrics['map']:.4f}")
    print(f"{'Player mAP@0.5':<25} {'0.0000':<15} {player_map_final:.4f}{'':<11} ✅ +{player_map_final:.4f}")
    print(f"{'Player Recall@0.5':<25} {'0.0000':<15} {player_recall_final:.4f}{'':<11} ✅ +{player_recall_final:.4f}")
    print(f"{'Ball mAP@0.5':<25} {'0.0000':<15} {ball_map_final:.4f}{'':<11} {'✅' if ball_map_final > 0 else '⚠️ '} {ball_map_final:.4f}")
    print(f"{'Ball Precision@0.5':<25} {'0.1400':<15} {ball_precision_final:.4f}{'':<11} {'✅' if ball_precision_final > 0.14 else '⚠️ '} {ball_precision_final - 0.14:+.4f}")
    print(f"{'Ball Recall@0.5':<25} {'0.5800':<15} {ball_recall_final:.4f}{'':<11} {'✅' if ball_recall_final > 0.50 else '⚠️ '} {ball_recall_final - 0.58:+.4f}")
    
    print("\n" + "=" * 70)
    print("PROGRESS ASSESSMENT")
    print("=" * 70)
    
    improvements = []
    if player_map_final > 0:
        improvements.append(f"✅ Player detection working (mAP: {player_map_final:.4f})")
    if player_recall_final > 0.50:
        improvements.append(f"✅ Player recall good ({player_recall_final:.4f})")
    if ball_map_final > 0:
        improvements.append(f"✅ Ball detection starting (mAP: {ball_map_final:.4f})")
    elif ball_recall_final > 0:
        improvements.append(f"⚠️  Ball recall present ({ball_recall_final:.4f}) but mAP still 0")
    
    if ball_precision_final > 0.14:
        improvements.append(f"✅ Ball precision improved ({ball_precision_final:.4f} vs 0.14)")
    elif ball_precision_final > 0:
        improvements.append(f"⚠️  Ball precision: {ball_precision_final:.4f} (needs more training)")
    
    for imp in improvements:
        print(f"  {imp}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if player_map_final > 0.20:
        print("✅ Player detection is working well - continue training")
    elif player_map_final > 0:
        print("⚠️  Player detection is improving but needs more epochs")
    else:
        print("❌ Player detection not working - check configuration")
    
    if ball_map_final > 0:
        print("✅ Ball detection is starting - continue training")
    elif ball_recall_final > 0:
        print("⚠️  Ball recall present but precision low - Focal Loss should help")
        print("   Continue training - ball mAP should improve")
    else:
        print("⚠️  Ball detection not yet appearing - may need more epochs")
        print("   This is normal - ball is harder to detect")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Continue training to epoch 10-20 to see ball mAP improve")
    print("2. Monitor ball precision - should improve with Focal Loss")
    print("3. Check training loss - should continue decreasing")
    print("4. Run full training (50-100 epochs) for best results")
    print("=" * 70)
    
    # Save checkpoint
    trainer.save_checkpoint(4, final_metrics['map'], is_best=False, is_interrupt=False, lightweight=True)
    print(f"\n✅ Checkpoint saved at epoch 5")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract and display epoch 5 training results"""
import re
import sys

log_file = "/tmp/training_epoch5.log"

def extract_results():
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch results
    epochs = {}
    for epoch_num in range(1, 6):
        pattern = f"EPOCH {epoch_num} RESULTS.*?(?=EPOCH {epoch_num+1}|FINAL PROGRESS|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            epoch_text = match.group(0)
            
            # Extract metrics
            metrics = {}
            metrics['player_map'] = float(re.search(r'Player.*?mAP@0\.5:\s+([\d.]+)', epoch_text).group(1))
            metrics['player_recall'] = float(re.search(r'Player.*?Recall:\s+([\d.]+)', epoch_text).group(1))
            metrics['ball_map'] = float(re.search(r'Ball.*?mAP@0\.5:\s+([\d.]+)', epoch_text).group(1))
            metrics['ball_recall'] = float(re.search(r'Ball.*?Recall:\s+([\d.]+)', epoch_text).group(1))
            metrics['overall_map'] = float(re.search(r'mAP:\s+([\d.]+)', epoch_text).group(1))
            
            # Training loss
            loss_match = re.search(r'Training Loss:\s+([\d.]+)', epoch_text)
            if loss_match:
                metrics['train_loss'] = float(loss_match.group(1))
            
            epochs[epoch_num] = metrics
    
    return epochs

if __name__ == "__main__":
    try:
        epochs = extract_results()
        
        print("=" * 70)
        print("EPOCH 5 TRAINING PROGRESS SUMMARY")
        print("=" * 70)
        
        if epochs:
            print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Player mAP':<12} {'Player Recall':<15} {'Ball mAP':<12} {'Overall mAP':<12}")
            print("-" * 70)
            
            for epoch_num in sorted(epochs.keys()):
                m = epochs[epoch_num]
                print(f"{epoch_num:<8} {m.get('train_loss', 0):<12.4f} {m['player_map']:<12.4f} {m['player_recall']:<15.4f} {m['ball_map']:<12.4f} {m['overall_map']:<12.4f}")
            
            # Show final epoch
            if 5 in epochs:
                final = epochs[5]
                print("\n" + "=" * 70)
                print("FINAL RESULTS (EPOCH 5)")
                print("=" * 70)
                print(f"Training Loss: {final.get('train_loss', 0):.4f}")
                print(f"Overall mAP: {final['overall_map']:.4f}")
                print(f"Player mAP@0.5: {final['player_map']:.4f}")
                print(f"Player Recall@0.5: {final['player_recall']:.4f}")
                print(f"Ball mAP@0.5: {final['ball_map']:.4f}")
                print(f"Ball Recall@0.5: {final['ball_recall']:.4f}")
        else:
            print("No complete epoch results found yet. Training may still be in progress.")
            print("Check /tmp/training_epoch5.log for current status.")
            
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        print("Training may not have started yet.")
    except Exception as e:
        print(f"Error extracting results: {e}")
        import traceback
        traceback.print_exc()

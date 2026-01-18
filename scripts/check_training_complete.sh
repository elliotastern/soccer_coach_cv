#!/bin/bash
# Check if training is complete and show results

LOG_FILE="/tmp/training_epoch5.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Training log not found. Training may not have started."
    exit 1
fi

if grep -q "FINAL PROGRESS EVALUATION" "$LOG_FILE"; then
    echo "✅ Training Complete!"
    echo ""
    echo "Final Results:"
    echo "=============="
    tail -100 "$LOG_FILE" | grep -A 50 "FINAL PROGRESS EVALUATION"
else
    echo "Training still in progress..."
    echo ""
    echo "Current Status:"
    echo "==============="
    tail -30 "$LOG_FILE" | grep -E "(EPOCH|Training Loss|mAP@0.5|Player|Ball)" | tail -10
    echo ""
    echo "To monitor: tail -f $LOG_FILE"
    echo "To check again: bash scripts/check_training_complete.sh"
fi

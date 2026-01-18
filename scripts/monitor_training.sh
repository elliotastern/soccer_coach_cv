#!/bin/bash
# Monitor training progress
LOG_FILE="/tmp/training_epoch5.log"

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

while true; do
    if [ -f "$LOG_FILE" ]; then
        # Get current epoch
        CURRENT_EPOCH=$(grep -o "EPOCH [0-9]/5" "$LOG_FILE" | tail -1)
        
        # Get latest metrics
        LATEST_METRICS=$(tail -50 "$LOG_FILE" | grep -A 10 "EPOCH.*RESULTS" | tail -15)
        
        clear
        echo "=========================================="
        echo "TRAINING PROGRESS MONITOR"
        echo "=========================================="
        echo "Current: $CURRENT_EPOCH"
        echo ""
        echo "Latest Results:"
        echo "$LATEST_METRICS"
        echo ""
        echo "Full log: tail -f $LOG_FILE"
        echo "=========================================="
        
        # Check if complete
        if grep -q "FINAL PROGRESS EVALUATION" "$LOG_FILE"; then
            echo ""
            echo "✅ TRAINING COMPLETE!"
            break
        fi
    fi
    sleep 5
done

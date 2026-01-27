#!/bin/bash
# Regenerate results with enhanced touchline detection for y-axis accuracy

VIDEO="data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
MODEL="models/rf_detr_soccertrack/checkpoint_best_ema.pth"
OUTPUT="output/video_37CAE053_touchline_enhanced"

echo "🔄 Regenerating results with enhanced touchline detection..."
echo "   Video: $VIDEO"
echo "   Model: $MODEL"
echo "   Output: $OUTPUT"
echo ""

python scripts/process_video_pipeline.py \
    "$VIDEO" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max-frames 50 \
    --confidence 0.3

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Results generated successfully!"
    echo "   Results: $OUTPUT/frame_data.json"
    echo ""
    # Create results.json symlink for validation script compatibility
    if [ -f "$OUTPUT/frame_data.json" ] && [ ! -f "$OUTPUT/results.json" ]; then
        ln -s frame_data.json "$OUTPUT/results.json"
    fi
    echo "📊 Running validation..."
    python scripts/validate_results.py \
        "$OUTPUT/results.json" \
        "$VIDEO" \
        --output "$OUTPUT/validation" \
        --num-frames 50
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Validation complete!"
        echo "   Viewer: $OUTPUT/validation/validation_viewer.html"
    fi
else
    echo "❌ Failed to generate results"
    exit 1
fi

#!/usr/bin/env python3
"""
Create a side-by-side comparison viewer for predictions from different epochs
"""
import json
from pathlib import Path
import argparse


def create_comparison_html(epoch1_dir: Path, epoch1_num: int, epoch2_dir: Path, epoch2_num: int, output_path: Path):
    """Create HTML viewer comparing two epochs side by side"""
    
    # Load predictions from both epochs
    pred1_path = epoch1_dir / "predictions.json"
    pred2_path = epoch2_dir / "predictions.json"
    
    with open(pred1_path) as f:
        pred1 = json.load(f)
    with open(pred2_path) as f:
        pred2 = json.load(f)
    
    # Match frames by original_frame_idx
    pred1_by_idx = {p['original_frame_idx']: p for p in pred1}
    pred2_by_idx = {p['original_frame_idx']: p for p in pred2}
    
    common_indices = sorted(set(pred1_by_idx.keys()) & set(pred2_by_idx.keys()))
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RF-DETR Epoch Comparison: {epoch1_num} vs {epoch2_num}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .epoch-section {{
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .epoch-header {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4CAF50;
        }}
        .epoch1-header {{
            color: #2196F3;
        }}
        .epoch2-header {{
            color: #FF9800;
        }}
        .frame-item {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
        }}
        .frame-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .frame-info {{
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }}
        .detection-count {{
            font-weight: bold;
            color: #4CAF50;
        }}
        .comparison-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .frame-header {{
            grid-column: 1 / -1;
            background-color: #333;
            color: white;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>RF-DETR Epoch Comparison: Epoch {epoch1_num} vs Epoch {epoch2_num}</h1>
    
    <div class="summary">
        <h2>📊 Comparison Summary</h2>
        <p><strong>Epoch {epoch1_num}:</strong> {len(pred1)} frames, {sum(p['num_detections'] for p in pred1)} total detections, {sum(p['num_detections'] for p in pred1) / len(pred1):.1f} avg/frame</p>
        <p><strong>Epoch {epoch2_num}:</strong> {len(pred2)} frames, {sum(p['num_detections'] for p in pred2)} total detections, {sum(p['num_detections'] for p in pred2) / len(pred2):.1f} avg/frame</p>
        <p><strong>Common Frames:</strong> {len(common_indices)}</p>
    </div>

    <h2>🎯 Side-by-Side Frame Comparisons</h2>
"""
    
    for frame_idx in common_indices:
        p1 = pred1_by_idx[frame_idx]
        p2 = pred2_by_idx[frame_idx]
        
        minutes1 = int(p1['timestamp'] // 60)
        seconds1 = int(p1['timestamp'] % 60)
        time_str1 = f"{minutes1:02d}:{seconds1:02d} ({p1['timestamp']:.1f}s)"
        
        minutes2 = int(p2['timestamp'] // 60)
        seconds2 = int(p2['timestamp'] % 60)
        time_str2 = f"{minutes2:02d}:{seconds2:02d} ({p2['timestamp']:.1f}s)"
        
        # Fix image paths - handle different directory structures
        # Epoch 62: "temp_soccerchallenge_predictions/prediction_01.jpg"
        # Epoch 99: "predictions/prediction_01.jpg" (relative to epoch2_dir)
        img1_path = p1['prediction_image_path']
        if img1_path.startswith(epoch1_dir.name + '/'):
            # Already has directory prefix
            pass
        elif '/' in img1_path:
            # Has subdirectory but not the main dir
            img1_path = f"{epoch1_dir.name}/{img1_path}"
        else:
            # Just filename
            img1_path = f"{epoch1_dir.name}/{img1_path}"
        
        img2_path = p2['prediction_image_path']
        if img2_path.startswith(epoch2_dir.name + '/'):
            # Already has directory prefix
            pass
        elif '/' in img2_path:
            # Has subdirectory but not the main dir
            img2_path = f"{epoch2_dir.name}/{img2_path}"
        else:
            # Just filename
            img2_path = f"{epoch2_dir.name}/{img2_path}"
        
        html_content += f"""
    <div class="frame-item">
        <div class="frame-header">
            Frame Index: {frame_idx} | Time: {time_str1}
        </div>
        <div class="comparison-row">
            <div class="epoch-section">
                <div class="epoch-header epoch1-header">Epoch {epoch1_num}</div>
                <img src="../{img1_path}" alt="Epoch {epoch1_num}">
                <div class="frame-info">
                    <span class="detection-count">Detections: {p1['num_detections']}</span>
                </div>
            </div>
            <div class="epoch-section">
                <div class="epoch-header epoch2-header">Epoch {epoch2_num}</div>
                <img src="../{img2_path}" alt="Epoch {epoch2_num}">
                <div class="frame-info">
                    <span class="detection-count">Detections: {p2['num_detections']}</span>
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Comparison viewer created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison viewer for two epochs")
    parser.add_argument(
        "--epoch1-dir",
        type=str,
        default="temp_soccerchallenge_predictions",
        help="Directory with epoch 1 predictions"
    )
    parser.add_argument(
        "--epoch1",
        type=int,
        default=62,
        help="Epoch 1 number"
    )
    parser.add_argument(
        "--epoch2-dir",
        type=str,
        default="temp_soccerchallenge_predictions_epoch99",
        help="Directory with epoch 2 predictions"
    )
    parser.add_argument(
        "--epoch2",
        type=int,
        default=99,
        help="Epoch 2 number"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="temp_soccerchallenge_predictions_comparison/epoch_comparison.html",
        help="Output HTML file path"
    )
    
    args = parser.parse_args()
    
    epoch1_dir = Path(args.epoch1_dir)
    epoch2_dir = Path(args.epoch2_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Creating Epoch Comparison Viewer")
    print("=" * 70)
    print(f"Epoch {args.epoch1}: {epoch1_dir}")
    print(f"Epoch {args.epoch2}: {epoch2_dir}")
    print(f"Output: {output_path}")
    print()
    
    create_comparison_html(epoch1_dir, args.epoch1, epoch2_dir, args.epoch2, output_path)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate 100 frames with ball annotations from a COCO dataset.
Generates HTML with toggleable bounding boxes.
"""
import json
import base64
from pathlib import Path
from typing import List, Dict
from PIL import Image
import io


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotation file."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def get_images_with_balls(coco_data: Dict) -> List[Dict]:
    """Get images that have ball annotations."""
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Find ball category ID
    ball_category_id = None
    for cat_id, cat_name in categories.items():
        if cat_name.lower() == 'ball':
            ball_category_id = cat_id
            break
    
    if ball_category_id is None:
        raise ValueError("Ball category not found in annotations")
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == ball_category_id:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann['bbox'])
    
    # Get images with balls
    images = {img['id']: img for img in coco_data['images']}
    images_with_balls = []
    
    for img_id in sorted(image_annotations.keys()):
        if img_id in images:
            images_with_balls.append({
                'image': images[img_id],
                'bboxes': image_annotations[img_id]
            })
    
    return images_with_balls


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 string."""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            img = Image.open(io.BytesIO(img_data))
            # Resize if too large (max 1920px width)
            max_width = 1920
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""


def generate_html(images_data: List[Dict], annotation_file: str, output_path: Path):
    """Generate HTML with toggleable bounding boxes."""
    
    annotation_name = Path(annotation_file).name
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ball Validation - {annotation_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .header h1 {{
            color: white;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 14px;
        }}
        
        .controls {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .toggle-btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .toggle-btn:hover {{
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .toggle-btn.off {{
            background: #ff9800;
        }}
        
        .toggle-btn.off:hover {{
            background: #f57c00;
        }}
        
        .stats {{
            color: #b0b0b0;
            font-size: 14px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        .frame-container {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }}
        
        .frame-container:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }}
        
        .image-wrapper {{
            position: relative;
            width: 100%;
            margin-bottom: 10px;
            border-radius: 6px;
            overflow: hidden;
            background: #1a1a1a;
        }}
        
        .image-wrapper img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .bbox-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        
        .bbox-overlay.hidden {{
            display: none;
        }}
        
        .bbox {{
            position: absolute;
            border: 3px solid #FFC107;
            background: rgba(255, 193, 7, 0.3);
            box-sizing: border-box;
        }}
        
        .bbox-label {{
            position: absolute;
            top: -20px;
            left: 0;
            background: rgba(0, 0, 0, 0.8);
            color: #FFC107;
            padding: 2px 6px;
            font-size: 12px;
            font-weight: bold;
            border-radius: 3px;
            white-space: nowrap;
        }}
        
        .frame-info {{
            color: #b0b0b0;
            font-size: 13px;
            margin-top: 8px;
        }}
        
        .frame-info strong {{
            color: #FFC107;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ Ball Validation - 100 Samples</h1>
        <p>Dataset: {annotation_name}</p>
        <p>Total frames with balls: {len(images_data)}</p>
    </div>
    
    <div class="controls">
        <button class="toggle-btn" id="toggleBtn" onclick="toggleBoxes()">Hide Boxes</button>
        <div class="stats">
            Showing {len(images_data)} frames with ball annotations
        </div>
    </div>
    
    <div class="grid">
"""
    
    for idx, img_data in enumerate(images_data):
        image_info = img_data['image']
        bboxes = img_data['bboxes']
        
        # Get image path
        annotation_dir = Path(annotation_file).parent
        image_path = annotation_dir / image_info['file_name']
        
        # Try alternative paths if image not found
        if not image_path.exists():
            # Try images subdirectory
            images_dir = annotation_dir / 'images'
            if images_dir.exists():
                image_path = images_dir / image_info['file_name']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Convert image to base64
        img_base64 = image_to_base64(image_path)
        if not img_base64:
            continue
        
        # Calculate bbox positions (relative to image)
        img_width = image_info['width']
        img_height = image_info['height']
        
        bbox_html = ""
        for bbox in bboxes:
            # COCO format: [x, y, width, height]
            x, y, w, h = bbox
            x_percent = (x / img_width) * 100
            y_percent = (y / img_height) * 100
            w_percent = (w / img_width) * 100
            h_percent = (h / img_height) * 100
            
            bbox_html += f"""
            <div class="bbox" style="left: {x_percent}%; top: {y_percent}%; width: {w_percent}%; height: {h_percent}%;">
                <div class="bbox-label">ball</div>
            </div>"""
        
        html_content += f"""
        <div class="frame-container">
            <div class="image-wrapper">
                <img src="{img_base64}" alt="Frame {idx + 1}">
                <div class="bbox-overlay" id="overlay-{idx}">
                    {bbox_html}
                </div>
            </div>
            <div class="frame-info">
                <strong>Frame {idx + 1}:</strong> {image_info['file_name']} | 
                <strong>{len(bboxes)}</strong> ball(s) | 
                Size: {img_width}x{img_height}
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        let boxesVisible = true;
        
        function toggleBoxes() {
            boxesVisible = !boxesVisible;
            const overlays = document.querySelectorAll('.bbox-overlay');
            const btn = document.getElementById('toggleBtn');
            
            overlays.forEach(overlay => {
                if (boxesVisible) {
                    overlay.classList.remove('hidden');
                } else {
                    overlay.classList.add('hidden');
                }
            });
            
            if (boxesVisible) {
                btn.textContent = 'Hide Boxes';
                btn.classList.remove('off');
            } else {
                btn.textContent = 'Show Boxes';
                btn.classList.add('off');
            }
        }
        
        // Keyboard shortcut: 'H' to toggle
        document.addEventListener('keydown', function(event) {
            if (event.key === 'h' || event.key === 'H') {
                toggleBoxes();
            }
        });
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated HTML: {output_path}")


def main():
    """Main function to validate 100 frames."""
    annotation_file = "/workspace/soccer_coach_cv/data/raw/real_data/Validation images OFFICIAL/valid/_annotations.coco.json"
    annotation_path = Path(annotation_file)
    
    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return
    
    print(f"📋 Loading annotations from: {annotation_file}")
    coco_data = load_coco_annotations(annotation_file)
    
    print("🔍 Finding images with ball annotations...")
    images_with_balls = get_images_with_balls(coco_data)
    
    print(f"📊 Found {len(images_with_balls)} images with ball annotations")
    
    # Select first 100 samples
    samples = images_with_balls[:100]
    print(f"✅ Selected {len(samples)} samples for validation")
    
    # Generate output filename
    annotation_name = annotation_path.stem
    if annotation_name.startswith('_'):
        annotation_name = annotation_name[1:]
    output_html = annotation_path.parent / f"9_validate_100_frames_{annotation_name}.html"
    
    print(f"🎨 Generating HTML visualization...")
    generate_html(samples, annotation_file, output_html)
    
    print(f"\n✅ Done! Open {output_html} in your browser to view the validation.")


if __name__ == "__main__":
    main()

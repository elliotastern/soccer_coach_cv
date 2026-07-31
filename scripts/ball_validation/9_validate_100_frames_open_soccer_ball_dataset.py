#!/usr/bin/env python3
"""
Validate 100 frames with ball annotations from the Open Soccer Ball Dataset.
Uses Pascal VOC XML format annotations.
Generates HTML with toggleable bounding boxes.
"""
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from typing import List, Dict
from PIL import Image
import io


def parse_voc_xml(xml_path: Path) -> Dict:
    """Parse Pascal VOC XML annotation file."""
    try:
        # Try parsing normally first
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        # If parsing fails, try to fix common XML issues
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Fix common issues: malformed <path> tags
        import re
        # Fix <path><path></path> to <path>Unknown</path>
        content = re.sub(r'<path><path></path>', '<path>Unknown</path>', content)
        content = re.sub(r'<path></path>', '<path>Unknown</path>', content)
        # Fix any remaining nested path tags
        content = re.sub(r'<path>.*?<path>.*?</path>.*?</path>', '<path>Unknown</path>', content, flags=re.DOTALL)
        
        try:
            root = ET.fromstring(content)
        except Exception as parse_err:
            # Last resort: try to extract just the essential parts
            print(f"Warning: Could not fully parse {xml_path}, attempting recovery...")
            raise ValueError(f"Could not parse XML file: {parse_err}")
    
    filename = root.find('filename')
    if filename is None or filename.text is None:
        raise ValueError("No filename found in XML")
    filename = filename.text
    
    size = root.find('size')
    if size is None:
        raise ValueError("No size found in XML")
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name')
        if name is None or name.text is None:
            continue
        if name.text.lower() == 'ball':
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'bboxes': bboxes
    }


def get_images_with_balls(annotations_dir: Path, images_dir: Path) -> List[Dict]:
    """Get all images that have ball annotations."""
    images_with_balls = []
    
    # Get all XML files
    xml_files = list(annotations_dir.glob('*.xml'))
    
    for xml_file in sorted(xml_files):
        try:
            annotation_data = parse_voc_xml(xml_file)
            
            if len(annotation_data['bboxes']) > 0:
                # Find corresponding image
                image_path = images_dir / annotation_data['filename']
                
                if image_path.exists():
                    images_with_balls.append({
                        'image_path': image_path,
                        'filename': annotation_data['filename'],
                        'width': annotation_data['width'],
                        'height': annotation_data['height'],
                        'bboxes': annotation_data['bboxes']
                    })
        except Exception as e:
            print(f"Warning: Error parsing {xml_file}: {e}")
            continue
    
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


def generate_html(images_data: List[Dict], dataset_name: str, output_path: Path):
    """Generate HTML with toggleable bounding boxes."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ball Validation - {dataset_name}</title>
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
        <h1>⚽ Ball Validation - Open Soccer Ball Dataset</h1>
        <p>Dataset: {dataset_name}</p>
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
        image_path = img_data['image_path']
        filename = img_data['filename']
        bboxes = img_data['bboxes']
        img_width = img_data['width']
        img_height = img_data['height']
        
        # Convert image to base64
        img_base64 = image_to_base64(image_path)
        if not img_base64:
            continue
        
        # Calculate bbox positions (Pascal VOC format: xmin, ymin, xmax, ymax)
        bbox_html = ""
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            x_percent = (xmin / img_width) * 100
            y_percent = (ymin / img_height) * 100
            w_percent = ((xmax - xmin) / img_width) * 100
            h_percent = ((ymax - ymin) / img_height) * 100
            
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
                <strong>Frame {idx + 1}:</strong> {filename} | 
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
    """Main function to validate frames."""
    base_dir = Path("/workspace/soccer_coach_cv/data/raw/Open Soccer Ball Dataset")
    
    # Process training set
    training_annotations = base_dir / "training" / "training" / "annotations"
    training_images = base_dir / "training" / "training" / "images"
    
    # Process test set
    test_annotations = base_dir / "test" / "ball" / "annotations"
    test_images = base_dir / "test" / "ball" / "img"
    
    all_results = {}
    
    # Process training
    if training_annotations.exists() and training_images.exists():
        print(f"📋 Processing training set...")
        print(f"   Annotations: {training_annotations}")
        print(f"   Images: {training_images}")
        
        images_with_balls = get_images_with_balls(training_annotations, training_images)
        print(f"📊 Found {len(images_with_balls)} images with ball annotations in training")
        
        # Select first 100 samples
        samples = images_with_balls[:100]
        print(f"✅ Selected {len(samples)} samples from training for validation")
        
        all_results['training'] = {
            'samples': samples,
            'total': len(images_with_balls)
        }
    
    # Process test
    if test_annotations.exists() and test_images.exists():
        print(f"\n📋 Processing test set...")
        print(f"   Annotations: {test_annotations}")
        print(f"   Images: {test_images}")
        
        images_with_balls = get_images_with_balls(test_annotations, test_images)
        print(f"📊 Found {len(images_with_balls)} images with ball annotations in test")
        
        # Select first 100 samples (or all if less)
        samples = images_with_balls[:100]
        print(f"✅ Selected {len(samples)} samples from test for validation")
        
        all_results['test'] = {
            'samples': samples,
            'total': len(images_with_balls)
        }
    
    # Generate HTML for each split
    for split_name, data in all_results.items():
        output_html = base_dir / f"9_validate_100_frames_{split_name}.html"
        print(f"\n🎨 Generating HTML visualization for {split_name}...")
        generate_html(data['samples'], f"Open Soccer Ball Dataset - {split_name}", output_html)
    
    print(f"\n✅ Done! Generated {len(all_results)} validation HTML files.")


if __name__ == "__main__":
    main()

"""
Convert YOLO format dataset to COCO format for RF-DETR training.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """Convert YOLO bbox to COCO bbox format."""
    class_id, x_center, y_center, width, height = yolo_bbox
    
    x_center_abs = float(x_center) * img_width
    y_center_abs = float(y_center) * img_height
    width_abs = float(width) * img_width
    height_abs = float(height) * img_height
    
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)
    
    return [x_min, y_min, width_abs, height_abs]


def load_yolo_annotation(txt_path):
    """Load YOLO format annotation file."""
    annotations = []
    if not os.path.exists(txt_path):
        return annotations
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    annotations.append([float(p) for p in parts])
    
    return annotations


def get_image_size(image_path):
    """Get image dimensions."""
    img = Image.open(image_path)
    return img.width, img.height


def convert_split(yolo_dir, output_dir, split_name, class_names):
    """Convert one split (train/test) from YOLO to COCO."""
    print(f"\n🔄 Converting {split_name} split...")
    
    yolo_path = Path(yolo_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    coco_data = {
        "info": {"description": f"SoccerSynth {split_name} dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for i, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })
    
    image_files = sorted(yolo_path.glob("*.png"))
    image_id = 1
    annotation_id = 1
    
    for img_file in image_files:
        txt_file = yolo_path / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            print(f"⚠️  Warning: No annotation for {img_file.name}, skipping")
            continue
        
        img_width, img_height = get_image_size(img_file)
        
        dest_image = images_dir / img_file.name
        shutil.copy2(img_file, dest_image)
        
        image_info = {
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_file.name
        }
        coco_data["images"].append(image_info)
        
        yolo_annotations = load_yolo_annotation(txt_file)
        
        for yolo_bbox in yolo_annotations:
            class_id = int(yolo_bbox[0])
            coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    annotation_file = annotations_dir / "annotations.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {len(coco_data['images'])} images")
    print(f"✅ Created {len(coco_data['annotations'])} annotations")
    print(f"✅ Saved to {annotation_file}")
    
    return len(coco_data['images']), len(coco_data['annotations'])


def main():
    """Main conversion function."""
    yolo_dataset_path = Path("/workspace/soccer_coach_cv/data/raw/SoccerSynth_sub")
    output_base = Path("/workspace/datasets")
    
    class_names = ["player", "ball"]
    
    print("🚀 Starting YOLO to COCO conversion...")
    print(f"📁 Source: {yolo_dataset_path}")
    print(f"📁 Destination: {output_base}")
    print(f"🏷️  Classes: {', '.join(class_names)}")
    
    train_images, train_annos = convert_split(
        yolo_dataset_path / "train",
        output_base / "train",
        "train",
        class_names
    )
    
    test_images, test_annos = convert_split(
        yolo_dataset_path / "test",
        output_base / "val",
        "val",
        class_names
    )
    
    print("\n📊 Conversion Summary:")
    print(f"   Train: {train_images} images, {train_annos} annotations")
    print(f"   Val:   {test_images} images, {test_annos} annotations")
    print(f"\n✅ Conversion complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug script to check DETR's raw label outputs
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.model import get_detr_model
from src.training.dataset import CocoDataset
from src.training.augmentation import get_val_transforms
from torch.utils.data import DataLoader
from src.training.collate import collate_fn

# Load model
checkpoint = torch.load('models/checkpoints/latest_checkpoint.pth', map_location='cpu', weights_only=False)
model = get_detr_model(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Get one sample
val_dataset = CocoDataset('/workspace/datasets/val', transforms=get_val_transforms({'resize': 1333}))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
images, targets = next(iter(val_loader))

print("Target labels:", targets[0]['labels'])
print("Target unique:", targets[0]['labels'].unique() if len(targets[0]['labels']) > 0 else "none")

# Run inference and check raw DETR output
with torch.no_grad():
    # Access the underlying DETR model
    detr_model = model.detr_model
    processor = model.processor
    
    # Process image
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = images[0]
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    pil_img = TF.to_pil_image(img_denorm)
    
    # Get DETR inputs
    inputs = processor(images=pil_img, return_tensors="pt")
    
    # Get raw DETR output
    outputs = detr_model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.0
    )[0]
    
    print("\nRaw DETR Output:")
    print("  Labels:", results['labels'])
    print("  Unique labels:", torch.unique(results['labels']).tolist())
    print("  Label distribution:", torch.bincount(results['labels'].long()).tolist())
    print("  Scores range:", results['scores'].min().item(), "to", results['scores'].max().item())
    print("  Number of predictions:", len(results['labels']))
    
    # Check what happens after our conversion
    raw_labels = results['labels']
    mask = raw_labels > 0
    print(f"\nAfter filtering background (label > 0):")
    print(f"  Valid predictions: {mask.sum().item()} out of {len(raw_labels)}")
    if mask.any():
        filtered_labels = raw_labels[mask] - 1
        print(f"  Converted labels: {filtered_labels.unique().tolist()}")
        print(f"  Converted label distribution: {torch.bincount(filtered_labels.long()).tolist()}")
    else:
        print("  No valid predictions after filtering!")

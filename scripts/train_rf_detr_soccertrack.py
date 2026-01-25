#!/usr/bin/env python3
"""
Train RF-DETR on SoccerTrack wide_view dataset for person detection
Uses RF-DETR out-of-the-box training API with custom player-specific augmentations
"""
from rfdetr import RFDETRMedium
from pathlib import Path
import argparse
import sys
import os

# Add custom augmentation path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Monkey-patch RF-DETR's make_coco_transforms to use our custom augmentations
from rfdetr.datasets import coco as rfdetr_coco
from src.training.augmentation import make_rfdetr_custom_transforms

# Store original function
_original_make_coco_transforms = rfdetr_coco.make_coco_transforms
_original_make_coco_transforms_square = rfdetr_coco.make_coco_transforms_square_div_64


def train_rf_detr_soccertrack(
    dataset_dir: str = "datasets/rf_detr_soccertrack",
    output_dir: str = "models/rf_detr_soccertrack",
    epochs: int = 100,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    resolution: int = 672,
    use_custom_augmentations: bool = True
):
    """
    Train RF-DETR on SoccerTrack dataset
    
    Args:
        dataset_dir: Path to dataset (should have train/ and valid/ folders)
        output_dir: Where to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        grad_accum_steps: Gradient accumulation steps
        lr: Learning rate
        resolution: Input image resolution (must be divisible by 56)
    """
    print("=" * 70)
    print("TRAINING RF-DETR ON SOCCERTRACK DATASET")
    print("=" * 70)
    print()
    
    dataset_path = Path(dataset_dir)
    print(f"Dataset directory: {dataset_path}")
    print(f"  Train annotations: {dataset_path / 'train' / '_annotations.coco.json'}")
    print(f"  Valid annotations: {dataset_path / 'valid' / '_annotations.coco.json'}")
    print()
    
    # Check dataset exists
    train_ann = dataset_path / "train" / "_annotations.coco.json"
    valid_ann = dataset_path / "valid" / "_annotations.coco.json"
    
    if not train_ann.exists():
        raise FileNotFoundError(f"Train annotations not found: {train_ann}")
    if not valid_ann.exists():
        raise FileNotFoundError(f"Valid annotations not found: {valid_ann}")
    
    print("Initializing RF-DETR Medium model...")
    model = RFDETRMedium()
    print("✅ Model initialized")
    print()
    
    # Apply custom augmentations if requested
    if use_custom_augmentations:
        print("🔧 Applying custom player-specific augmentations:")
        print("   - MixUp: 0.2 (occlusion handling)")
        print("   - Copy-Paste: 0.3 (player pasting)")
        print("   - Erasing: 0.4 (occlusion simulation)")
        print("   - Mosaic: 1.0 (always on, crowd context)")
        print("   - HSV: h=0.01, s=0.6, v=0.4 (team color protection)")
        print("   - Geometric: shear=5.0, perspective=0.001, scale=0.8")
        print("   - Rotation: OFF (degrees=0.0)")
        print()
        
        # Monkey-patch the transform functions
        def custom_make_transforms(image_set, resolution, multi_scale=False, 
                                   expanded_scales=False, skip_random_resize=False,
                                   patch_size=16, num_windows=4):
            # We'll need to get the dataset later, so we create a wrapper
            # that will be called with the dataset
            return None  # Return None to signal we'll set it later
        
        def custom_make_transforms_square(image_set, resolution, multi_scale=False,
                                         expanded_scales=False, skip_random_resize=False,
                                         patch_size=16, num_windows=4):
            return None
        
        # Store the custom function
        rfdetr_coco._custom_transform_fn = make_rfdetr_custom_transforms
        
        # Patch the build functions to use custom transforms
        from rfdetr.datasets.coco import build_roboflow_from_coco as original_build_roboflow
        
        def patched_build_roboflow_from_coco(image_set, args, resolution):
            """Patched build function that uses custom transforms"""
            from pathlib import Path
            from rfdetr.datasets.coco import CocoDetection
            
            root = Path(args.dataset_dir)
            PATHS = {
                "train": (root / "train", root / "train" / "_annotations.coco.json"),
                "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
                "test": (root / "test", root / "test" / "_annotations.coco.json"),
            }
            
            img_folder, ann_file = PATHS[image_set.split("_")[0]]
            include_masks = getattr(args, "segmentation_head", False)
            
            # Create transforms first (without dataset for now)
            transforms = make_rfdetr_custom_transforms(
                image_set.split("_")[0],
                resolution,
                dataset=None  # Will set after dataset creation
            )
            
            # Create dataset with transforms
            dataset = CocoDetection(img_folder, ann_file, transforms=transforms, include_masks=include_masks)
            
            # Now update transforms' dataset references for MixUp/Mosaic/CopyPaste
            if image_set.split("_")[0] == 'train' and hasattr(transforms, 'transforms'):
                for transform in transforms.transforms:
                    if hasattr(transform, 'dataset'):
                        transform.dataset = dataset
            
            return dataset
        
        # Replace the build function
        rfdetr_coco.build_roboflow_from_coco = patched_build_roboflow_from_coco
        print("✅ Custom augmentations enabled")
        print()
    
    print("Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Resolution: {resolution}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Check for existing checkpoint to resume from
    checkpoint_path = Path(output_dir) / "checkpoint.pth"
    resume = None
    if checkpoint_path.exists():
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Resuming training from checkpoint...")
        resume = str(checkpoint_path)
    
    # Train the model
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        resolution=resolution,
        output_dir=output_dir,
        resume=resume
    )
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Best model: {Path(output_dir) / 'checkpoint_best_total.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF-DETR on SoccerTrack dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/rf_detr_soccertrack",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/rf_detr_soccertrack",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=672,
        help="Input image resolution (must be divisible by 56)"
    )
    
    args = parser.parse_args()
    
    train_rf_detr_soccertrack(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution
    )

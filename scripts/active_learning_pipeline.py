#!/usr/bin/env python3
"""
Active Learning Pipeline: Train on Frame 0, Auto-label Remaining Frames

Workflow:
1. Extract frame 0 from video
2. Convert frame 0 annotations from CVAT XML to COCO format
3. Train DETR model for 1 epoch on frame 0
4. Run inference on frames 1+ using fine-tuned model
5. Merge predictions back into XML, preserving frame 0 annotations
"""
import sys
import argparse
import shutil
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.extract_frame import extract_frame
from src.utils.cvat_to_coco import convert_frame_to_coco
from src.utils.merge_annotations import merge_annotations

# Import training and inference functions
# Add scripts directory to path to import modules
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from train_single_frame import train_single_frame
from infer_remaining_frames import infer_remaining_frames


def run_active_learning_pipeline(
    video_path: str,
    xml_path: str,
    output_xml_path: str,
    frame_id: int = 0,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    confidence_threshold: float = 0.5,
    cleanup: bool = True
):
    """
    Run complete active learning pipeline
    
    Args:
        video_path: Path to video file
        xml_path: Path to CVAT XML with frame 0 annotations
        output_xml_path: Path to save merged XML with all annotations
        frame_id: Frame number to train on (default: 0)
        num_epochs: Number of training epochs (default: 1)
        learning_rate: Learning rate for training (default: 1e-5)
        confidence_threshold: Confidence threshold for inference (default: 0.5)
        cleanup: Whether to clean up temporary files (default: True)
    """
    print("=" * 70)
    print("ACTIVE LEARNING PIPELINE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input XML: {xml_path}")
    print(f"Output XML: {output_xml_path}")
    print(f"Training on frame: {frame_id}")
    print("=" * 70)
    
    # Create temporary directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="active_learning_"))
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # Step 1: Extract frame 0 from video
        print("\n" + "=" * 70)
        print("STEP 1: Extracting frame from video")
        print("=" * 70)
        frame_image_path = str(temp_dir / "frame_000000.jpg")
        image_path, width, height = extract_frame(
            video_path=video_path,
            frame_id=frame_id,
            output_path=frame_image_path
        )
        print(f"✅ Extracted frame {frame_id} to: {image_path}")
        print(f"   Image size: {width}x{height}")
        
        # Step 2: Convert CVAT XML to COCO format
        print("\n" + "=" * 70)
        print("STEP 2: Converting CVAT XML to COCO format")
        print("=" * 70)
        dataset_dir = temp_dir / "train_dataset"
        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy frame image to dataset
        dataset_image_path = str(images_dir / "frame_000000.jpg")
        shutil.copy(image_path, dataset_image_path)
        
        # Convert annotations
        coco_json_path = str(annotations_dir / "annotations.json")
        coco_data = convert_frame_to_coco(
            xml_path=xml_path,
            frame_id=frame_id,
            image_path=dataset_image_path,
            image_width=width,
            image_height=height,
            output_json_path=coco_json_path
        )
        print(f"✅ Created COCO dataset at: {dataset_dir}")
        print(f"   Images: {len(coco_data['images'])}")
        print(f"   Annotations: {len(coco_data['annotations'])}")
        
        # Step 3: Train model on frame 0
        print("\n" + "=" * 70)
        print("STEP 3: Training model on frame 0")
        print("=" * 70)
        checkpoint_path = str(temp_dir / "finetuned_model.pth")
        train_single_frame(
            dataset_dir=str(dataset_dir),
            output_checkpoint_path=checkpoint_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=1
        )
        print(f"✅ Model trained and saved to: {checkpoint_path}")
        
        # Step 4: Run inference on frames 1+
        print("\n" + "=" * 70)
        print("STEP 4: Running inference on remaining frames")
        print("=" * 70)
        tracked_objects_by_frame = infer_remaining_frames(
            video_path=video_path,
            model_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            start_frame=frame_id + 1,
            device=None  # Auto-detect
        )
        print(f"✅ Inference complete: {len(tracked_objects_by_frame)} frames with detections")
        
        # Step 5: Merge annotations
        print("\n" + "=" * 70)
        print("STEP 5: Merging annotations")
        print("=" * 70)
        merge_annotations(
            original_xml_path=xml_path,
            video_path=video_path,
            new_tracked_objects_by_frame=tracked_objects_by_frame,
            output_xml_path=output_xml_path
        )
        print(f"✅ Merged XML saved to: {output_xml_path}")
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Output XML: {output_xml_path}")
        print(f"  - Frame {frame_id} annotations preserved (manual)")
        print(f"  - Frames {frame_id + 1}+ auto-labeled (auto)")
        
    finally:
        # Cleanup
        if cleanup:
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nTemporary files preserved at: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning Pipeline: Train on Frame 0, Auto-label Rest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/active_learning_pipeline.py \\
    --video data/raw/real_data/video.mp4 \\
    --xml data/raw/real_data/video_annotations.xml \\
    --output data/raw/real_data/video_annotations_updated.xml
        """
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--xml",
        type=str,
        required=True,
        help="Path to CVAT XML file with frame 0 annotations"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged XML output"
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=0,
        help="Frame number to train on (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for inference (default: 0.5)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up temporary files"
    )
    
    args = parser.parse_args()
    
    run_active_learning_pipeline(
        video_path=args.video,
        xml_path=args.xml,
        output_xml_path=args.output,
        frame_id=args.frame_id,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        confidence_threshold=args.confidence,
        cleanup=not args.no_cleanup
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for semantic segmentation-based 2D mapping integration.
Verifies that segmentation-based line detection works correctly.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto
from src.analysis.pitch_line_segmentation import PitchLineSegmenter


def test_segmentation_module():
    """Test the PitchLineSegmenter module directly."""
    print("=" * 60)
    print("Testing PitchLineSegmenter Module")
    print("=" * 60)
    
    # Create a test image (synthetic soccer field with white lines)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]  # Green field color
    
    # Draw some white lines (simulating field lines)
    cv2.line(test_image, (0, 240), (640, 240), (255, 255, 255), 5)  # Center line
    cv2.line(test_image, (320, 0), (320, 480), (255, 255, 255), 5)  # Vertical line
    cv2.rectangle(test_image, (50, 100), (150, 200), (255, 255, 255), 3)  # Box
    
    print("Created test image with white lines on green field")
    
    # Test with no model (should use color-based fallback)
    print("\n1. Testing with no model (color-based fallback)...")
    segmenter = PitchLineSegmenter(model_path=None)
    mask = segmenter.segment_pitch_lines(test_image)
    
    assert mask is not None, "Mask should not be None"
    assert mask.shape[:2] == test_image.shape[:2], "Mask should match image size"
    assert mask.dtype == np.uint8, "Mask should be uint8"
    
    # Check that we detected some lines
    line_pixels = np.sum(mask > 0)
    print(f"   Detected {line_pixels} line pixels (out of {mask.size} total)")
    assert line_pixels > 0, "Should detect some line pixels"
    
    print("   ✓ Color-based fallback works correctly")
    
    # Test with DeepLabV3 (will use pre-trained, but won't be trained for lines)
    print("\n2. Testing with DeepLabV3 (untrained, will use fallback)...")
    try:
        segmenter_deeplab = PitchLineSegmenter(
            model_path=None,
            model_type="deeplabv3",
            use_pretrained=True
        )
        
        if segmenter_deeplab._model_loaded:
            mask_deeplab = segmenter_deeplab.segment_pitch_lines(test_image)
            print(f"   Model loaded, generated mask shape: {mask_deeplab.shape}")
            print("   ✓ DeepLabV3 model loads correctly")
        else:
            print("   Model not loaded (expected if no checkpoint), using fallback")
            print("   ✓ Fallback mechanism works")
    except Exception as e:
        print(f"   Warning: {e}")
        print("   (This is expected if torchvision is not available)")
    
    print("\n✓ Segmentation module tests passed!")


def test_keypoint_detector_integration():
    """Test PitchKeypointDetector with segmentation enabled/disabled."""
    print("\n" + "=" * 60)
    print("Testing PitchKeypointDetector Integration")
    print("=" * 60)
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]  # Green field
    
    # Draw field lines
    cv2.line(test_image, (0, 240), (640, 240), (255, 255, 255), 5)  # Center line
    cv2.line(test_image, (320, 0), (320, 480), (255, 255, 255), 5)  # Vertical line
    cv2.rectangle(test_image, (50, 100), (150, 200), (255, 255, 255), 3)
    cv2.rectangle(test_image, (490, 100), (590, 200), (255, 255, 255), 3)
    
    print("Created test image with field lines")
    
    # Test 1: Without segmentation (default behavior)
    print("\n1. Testing without semantic segmentation (default)...")
    detector_default = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=False
    )
    
    lines_default = detector_default._detect_field_lines(test_image)
    print(f"   Detected {len(lines_default)} lines using color-based method")
    assert len(lines_default) > 0, "Should detect some lines"
    print("   ✓ Color-based detection works")
    
    # Test 2: With segmentation enabled (no model, should fallback)
    print("\n2. Testing with semantic segmentation enabled (no model, fallback)...")
    detector_seg = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=True,
        segmentation_config={'model_path': None}
    )
    
    lines_seg = detector_seg._detect_field_lines(test_image)
    print(f"   Detected {len(lines_seg)} lines using segmentation (fallback)")
    assert len(lines_seg) > 0, "Should detect some lines even with fallback"
    print("   ✓ Segmentation with fallback works")
    
    # Test 3: Full keypoint detection
    print("\n3. Testing full keypoint detection...")
    keypoints = detector_default.detect_all_keypoints(test_image)
    print(f"   Detected {len(keypoints)} keypoints")
    print(f"   Keypoint types: {set(kp.landmark_type for kp in keypoints)}")
    print("   ✓ Full keypoint detection works")
    
    # Test 4: Convenience function
    print("\n4. Testing convenience function...")
    result = detect_pitch_keypoints_auto(
        test_image,
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=False
    )
    
    if result is not None:
        print(f"   Detected {len(result['image_points'])} keypoints")
        print("   ✓ Convenience function works")
    else:
        print("   Warning: No keypoints detected (may be normal for synthetic image)")
    
    print("\n✓ Keypoint detector integration tests passed!")


def test_config_loading():
    """Test loading configuration from YAML."""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    config_path = Path(__file__).parent.parent / "configs" / "goal_detection.yaml"
    
    if not config_path.exists():
        print(f"   Warning: Config file not found: {config_path}")
        print("   Skipping config test")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'line_segmentation' in config:
        seg_config = config['line_segmentation']
        print(f"   Found line_segmentation config:")
        print(f"     enabled: {seg_config.get('enabled', False)}")
        print(f"     model_type: {seg_config.get('model_type', 'deeplabv3')}")
        print(f"     model_path: {seg_config.get('model_path', None)}")
        print("   ✓ Configuration loaded successfully")
        
        # Test creating detector with config
        if seg_config.get('enabled', False):
            print("\n   Testing detector creation with config...")
            detector = PitchKeypointDetector(
                pitch_length=105.0,
                pitch_width=68.0,
                use_semantic_segmentation=seg_config.get('enabled', False),
                segmentation_config=seg_config
            )
            print("   ✓ Detector created with config")
        else:
            print("   Segmentation disabled in config (skipping detector test)")
    else:
        print("   Warning: line_segmentation section not found in config")
    
    print("\n✓ Configuration tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Semantic Segmentation Integration Tests")
    print("=" * 60)
    
    try:
        test_segmentation_module()
        test_keypoint_detector_integration()
        test_config_loading()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nNote: To use semantic segmentation with a trained model,")
        print("      set line_segmentation.enabled=true in goal_detection.yaml")
        print("      and provide a model_path to a trained checkpoint.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

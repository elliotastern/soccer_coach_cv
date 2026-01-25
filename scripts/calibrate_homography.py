#!/usr/bin/env python3
"""
Interactive tool for manual homography calibration (keyframe initialization).

Allows user to select 4 pitch landmarks on a video frame and map them to
standard 105x68m pitch coordinates.
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Standard pitch dimensions (FIFA)
PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters

# Standard pitch landmarks (in meters, origin at center)
LANDMARKS = {
    "center_circle_center": (0.0, 0.0),
    "center_circle_top": (0.0, -9.15),
    "center_circle_bottom": (0.0, 9.15),
    "penalty_box_left_top": (-105.0/2 + 16.5, -20.16),
    "penalty_box_left_bottom": (-105.0/2 + 16.5, 20.16),
    "penalty_box_left_corner": (-105.0/2, 0.0),
    "penalty_box_right_top": (105.0/2 - 16.5, -20.16),
    "penalty_box_right_bottom": (105.0/2 - 16.5, 20.16),
    "penalty_box_right_corner": (105.0/2, 0.0),
    "corner_top_left": (-105.0/2, -68.0/2),
    "corner_top_right": (105.0/2, -68.0/2),
    "corner_bottom_left": (-105.0/2, 68.0/2),
    "corner_bottom_right": (105.0/2, 68.0/2),
    "center_top": (0.0, -68.0/2),
    "center_bottom": (0.0, 68.0/2),
}


class HomographyCalibrator:
    """Interactive homography calibration tool"""
    
    def __init__(self, image: np.ndarray, pitch_length: float = PITCH_LENGTH, 
                 pitch_width: float = PITCH_WIDTH):
        """
        Initialize calibrator
        
        Args:
            image: Keyframe image (BGR)
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
        """
        self.image = image.copy()
        self.display_image = image.copy()
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        # Selected points
        self.image_points = []  # Pixel coordinates
        self.pitch_points = []  # Pitch coordinates (meters)
        
        # Current selection state
        self.current_point_idx = 0
        self.selecting = False
        
        # Window name
        self.window_name = "Homography Calibration - Click 4 points"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_points) < 4:
                self.image_points.append((float(x), float(y)))
                # Draw point
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, f"P{len(self.image_points)}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.display_image)
                
                # Prompt for pitch coordinates
                if len(self.image_points) == 4:
                    self._prompt_pitch_coordinates()
    
    def _prompt_pitch_coordinates(self):
        """Prompt user to enter pitch coordinates for each point"""
        print("\n" + "="*60)
        print("Enter pitch coordinates for each point (in meters)")
        print("Origin (0, 0) is at center of pitch")
        print("X: -52.5 to +52.5 (left to right)")
        print("Y: -34 to +34 (top to bottom)")
        print("="*60)
        
        print("\nAvailable landmarks:")
        for i, (name, coords) in enumerate(list(LANDMARKS.items())[:8], 1):
            print(f"  {i}. {name}: {coords}")
        print("  9. Custom coordinates")
        
        self.pitch_points = []
        
        for i, img_pt in enumerate(self.image_points, 1):
            print(f"\nPoint {i} (pixel: {img_pt[0]:.1f}, {img_pt[1]:.1f}):")
            
            while True:
                try:
                    choice = input("Select landmark (1-9) or enter 'x,y': ").strip()
                    
                    if ',' in choice:
                        # Direct coordinate input
                        x, y = map(float, choice.split(','))
                        self.pitch_points.append((x, y))
                        break
                    elif choice.isdigit() and 1 <= int(choice) <= 8:
                        # Landmark selection
                        landmark_name = list(LANDMARKS.keys())[int(choice) - 1]
                        coords = LANDMARKS[landmark_name]
                        self.pitch_points.append(coords)
                        print(f"  Selected: {landmark_name} = {coords}")
                        break
                    elif choice == '9':
                        # Custom coordinates
                        coords_str = input("Enter x,y coordinates: ").strip()
                        x, y = map(float, coords_str.split(','))
                        self.pitch_points.append((x, y))
                        break
                    else:
                        print("Invalid input. Try again.")
                except (ValueError, IndexError) as e:
                    print(f"Invalid input: {e}. Try again.")
        
        print("\n" + "="*60)
        print("Selected correspondences:")
        for i, (img_pt, pitch_pt) in enumerate(zip(self.image_points, self.pitch_points), 1):
            print(f"  P{i}: Pixel ({img_pt[0]:.1f}, {img_pt[1]:.1f}) -> Pitch ({pitch_pt[0]:.2f}, {pitch_pt[1]:.2f})")
        print("="*60)
    
    def calibrate(self) -> Optional[np.ndarray]:
        """
        Perform calibration and return homography matrix
        
        Returns:
            Homography matrix (3x3) or None if calibration failed
        """
        if len(self.image_points) != 4 or len(self.pitch_points) != 4:
            print("Error: Need exactly 4 point correspondences")
            return None
        
        # Convert to numpy arrays
        image_pts = np.array(self.image_points, dtype=np.float32)
        pitch_pts = np.array(self.pitch_points, dtype=np.float32)
        
        # Estimate homography
        H, mask = cv2.findHomography(image_pts, pitch_pts, 
                                     method=cv2.RANSAC,
                                     ransacReprojThreshold=5.0)
        
        if H is None:
            print("Error: Homography estimation failed")
            return None
        
        # Calculate reprojection error
        errors = []
        for img_pt, pitch_pt in zip(self.image_points, self.pitch_points):
            # Transform image point to pitch
            pt_h = np.array([img_pt[0], img_pt[1], 1.0])
            transformed = H @ pt_h
            x_pred = transformed[0] / transformed[2]
            y_pred = transformed[1] / transformed[2]
            
            error = np.sqrt((x_pred - pitch_pt[0])**2 + (y_pred - pitch_pt[1])**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nCalibration Results:")
        print(f"  Mean reprojection error: {mean_error:.3f} meters")
        print(f"  Max reprojection error: {max_error:.3f} meters")
        
        if mean_error > 2.0:
            print("  Warning: High reprojection error. Consider re-calibrating.")
        
        return H
    
    def visualize(self, H: np.ndarray):
        """Visualize homography by projecting pitch lines onto image"""
        vis_image = self.image.copy()
        
        # Draw selected points
        for i, pt in enumerate(self.image_points, 1):
            cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
            cv2.putText(vis_image, f"P{i}", 
                       (int(pt[0]) + 10, int(pt[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Project pitch lines
        # Center line
        center_line_pitch = np.array([
            [0, -PITCH_WIDTH/2],
            [0, PITCH_WIDTH/2]
        ], dtype=np.float32)
        
        # Transform to image coordinates (inverse homography)
        H_inv = np.linalg.inv(H)
        center_line_img = cv2.perspectiveTransform(
            center_line_pitch.reshape(1, -1, 2), H_inv
        )[0]
        
        cv2.line(vis_image, 
                (int(center_line_img[0, 0]), int(center_line_img[0, 1])),
                (int(center_line_img[1, 0]), int(center_line_img[1, 1])),
                (255, 0, 0), 2)
        
        cv2.imshow(self.window_name, vis_image)
        print("\nPress any key to continue...")
        cv2.waitKey(0)


def calibrate_from_image(image_path: str, output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Calibrate homography from a single image
    
    Args:
        image_path: Path to keyframe image
        output_path: Optional path to save homography matrix (JSON)
    
    Returns:
        Homography matrix (3x3) or None
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    # Create calibrator
    calibrator = HomographyCalibrator(image)
    
    # Setup window and mouse callback
    cv2.namedWindow(calibrator.window_name)
    cv2.setMouseCallback(calibrator.window_name, calibrator.mouse_callback)
    
    # Display instructions
    print("="*60)
    print("HOMOGRAPHY CALIBRATION")
    print("="*60)
    print("Instructions:")
    print("1. Click 4 distinct points on the pitch (e.g., corners, penalty box)")
    print("2. After clicking 4 points, you'll be prompted for pitch coordinates")
    print("3. Select landmarks or enter custom coordinates")
    print("4. Press 'q' to quit, 'r' to reset")
    print("="*60)
    
    cv2.imshow(calibrator.window_name, calibrator.display_image)
    
    # Event loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            calibrator.image_points = []
            calibrator.pitch_points = []
            calibrator.display_image = image.copy()
            cv2.imshow(calibrator.window_name, calibrator.display_image)
        
        if len(calibrator.image_points) == 4 and len(calibrator.pitch_points) == 4:
            # Calibrate
            H = calibrator.calibrate()
            if H is not None:
                calibrator.visualize(H)
                
                # Save if output path provided
                if output_path:
                    save_homography(H, output_path, calibrator.image_points, calibrator.pitch_points)
                
                cv2.destroyAllWindows()
                return H
            break
    
    cv2.destroyAllWindows()
    return None


def calibrate_from_video(video_path: str, frame_number: int = 0, 
                        output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Calibrate homography from a video frame
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to use (default: 0)
        output_path: Optional path to save homography matrix
    
    Returns:
        Homography matrix (3x3) or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None
    
    return calibrate_from_image_frame(frame, output_path)


def calibrate_from_image_frame(frame: np.ndarray, output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Calibrate from numpy array frame"""
    calibrator = HomographyCalibrator(frame)
    
    cv2.namedWindow(calibrator.window_name)
    cv2.setMouseCallback(calibrator.window_name, calibrator.mouse_callback)
    
    print("="*60)
    print("HOMOGRAPHY CALIBRATION")
    print("="*60)
    print("Click 4 distinct points on the pitch")
    print("Press 'q' to quit, 'r' to reset")
    print("="*60)
    
    cv2.imshow(calibrator.window_name, calibrator.display_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            calibrator.image_points = []
            calibrator.pitch_points = []
            calibrator.display_image = frame.copy()
            cv2.imshow(calibrator.window_name, calibrator.display_image)
        
        if len(calibrator.image_points) == 4 and len(calibrator.pitch_points) == 4:
            H = calibrator.calibrate()
            if H is not None:
                calibrator.visualize(H)
                if output_path:
                    save_homography(H, output_path, calibrator.image_points, calibrator.pitch_points)
                cv2.destroyAllWindows()
                return H
            break
    
    cv2.destroyAllWindows()
    return None


def save_homography(H: np.ndarray, output_path: str, 
                   image_points: List[Tuple[float, float]],
                   pitch_points: List[Tuple[float, float]]):
    """
    Save homography matrix to JSON file
    
    Args:
        H: Homography matrix (3x3)
        output_path: Output file path
        image_points: Image point correspondences
        pitch_points: Pitch point correspondences
    """
    data = {
        "homography": H.tolist(),
        "image_points": image_points,
        "pitch_points": pitch_points,
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved homography to: {output_path}")


def load_homography(input_path: str) -> Optional[np.ndarray]:
    """
    Load homography matrix from JSON file
    
    Args:
        input_path: Input file path
    
    Returns:
        Homography matrix (3x3) or None
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    H = np.array(data['homography'], dtype=np.float32)
    return H


def main():
    parser = argparse.ArgumentParser(description="Calibrate homography for pitch mapping")
    parser.add_argument("input", type=str, help="Input image or video file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for homography")
    parser.add_argument("--frame", "-f", type=int, default=0, 
                       help="Frame number for video (default: 0)")
    parser.add_argument("--video", action="store_true", 
                       help="Treat input as video file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Determine if video or image
    is_video = args.video or input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
    if is_video:
        H = calibrate_from_video(str(input_path), args.frame, args.output)
    else:
        H = calibrate_from_image(str(input_path), args.output)
    
    if H is not None:
        print("\n✅ Calibration successful!")
        if args.output:
            print(f"   Homography saved to: {args.output}")
    else:
        print("\n❌ Calibration failed")


if __name__ == "__main__":
    main()

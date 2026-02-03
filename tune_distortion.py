import cv2
import numpy as np

# CONFIGURATION
VIDEO_PATH = '/workspace/soccer_coach_cv/data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4'
SCALE = 0.5  # Scale down for faster tuning


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Read one frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to load video")
        return

    try:
        cv2.namedWindow("x")
        cv2.destroyWindow("x")
    except cv2.error:
        cap.release()
        print("No display available. Run the web tuner instead:")
        print("  python tune_distortion_web.py")
        print("Then open http://localhost:8080 in your browser.")
        return

    h, w = frame.shape[:2]

    # Create Window
    cv2.namedWindow('Defish Tuner')
    cv2.resizeWindow('Defish Tuner', int(w * SCALE), int(h * SCALE))

    # Create Trackbars
    # We map 0-1000 sliders to float values (e.g., 500 = 0.0)
    cv2.createTrackbar('K1 (Radial)', 'Defish Tuner', 500, 1000, nothing)
    cv2.createTrackbar('K2 (Radial)', 'Defish Tuner', 500, 1000, nothing)
    cv2.createTrackbar('P1 (Tangential)', 'Defish Tuner', 500, 1000, nothing)
    cv2.createTrackbar('P2 (Tangential)', 'Defish Tuner', 500, 1000, nothing)

    # Camera Matrix (Estimate center of image)
    # [fx, 0, cx]
    # [0, fy, cy]
    # [0,  0,  1]
    # We assume fx = fy = width (approx for wide angle)
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]])

    while True:
        # Get Slider Values
        k1 = (cv2.getTrackbarPos('K1 (Radial)', 'Defish Tuner') - 500) / 1000.0
        k2 = (cv2.getTrackbarPos('K2 (Radial)', 'Defish Tuner') - 500) / 1000.0
        p1 = (cv2.getTrackbarPos('P1 (Tangential)', 'Defish Tuner') - 500) / 1000.0
        p2 = (cv2.getTrackbarPos('P2 (Tangential)', 'Defish Tuner') - 500) / 1000.0

        # Create Distortion Vector
        # [k1, k2, p1, p2, k3(0)]
        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float64)

        # Apply Undistort
        # getOptimalNewCameraMatrix allows us to keep the image valid
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

        undistorted_frame = cv2.undistort(frame, K, dist_coeffs, None, new_camera_matrix)

        # Draw a Reference Line (Horizontal and Vertical Center)
        # To help you see if lines are straight
        cv2.line(undistorted_frame, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 255), 2)
        cv2.line(undistorted_frame, (int(w / 2), 0), (int(w / 2), h), (0, 0, 255), 2)

        # Show Output
        cv2.imshow('Defish Tuner', cv2.resize(undistorted_frame, (0, 0), fx=SCALE, fy=SCALE))

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n--- SAVED PARAMETERS ---")
            print(f"k1: {k1}")
            print(f"k2: {k2}")
            print(f"p1: {p1}")
            print(f"p2: {p2}")
            print("------------------------\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

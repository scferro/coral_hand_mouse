# test.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
from pose.hand_pose import HandPoseEstimator, PoseConfig

def main():
    """
    Main demonstration script for hand pose estimation.
    Shows real-time hand pose tracking with coordinate frame visualization.
    """
    print("Initializing hand pose estimation system...")
    
    # Initialize MediaPipe hand detector
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,  # Track only one hand for stability
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("MediaPipe hand detector initialized")
    
    # Initialize pose estimator with custom configuration
    config = PoseConfig(
        focal_length=500.0,         # Approximate focal length for webcam
        principal_point=(320, 240), # Center of image (half resolution)
        hand_scale=0.08,           # 8cm hand size
        smoothing_factor=0.7,      # Strong smoothing for stability
        debug=True                 # Enable debug output
    )
    pose_estimator = HandPoseEstimator(config)
    print("Hand pose estimator initialized")
    
    # Initialize webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    # Set resolution (adjust if needed for your camera)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Create window for display
    window_name = 'Hand Pose Estimation'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    print("\nStarting hand pose estimation...")
    print("Place your hand in front of the camera")
    print("Press 'q' to quit, 'r' to reset tracking")
    
    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_update_interval = 1.0  # Update FPS every second
    last_fps_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time > fps_update_interval:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hand landmarks
            detection_result = detector.detect(mp_image)
            
            # If hand detected, estimate pose
            display_frame = frame.copy()
            if detection_result.hand_landmarks:
                # Estimate hand pose
                pose = pose_estimator.estimate_pose(
                    detection_result,
                    (frame.shape[1], frame.shape[0])
                )
                
                if pose:
                    # Visualize the pose
                    display_frame = pose_estimator.draw_pose(display_frame, pose)
            
            # Add FPS counter
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show the frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('r'):
                print("\nResetting tracking...")
                pose_estimator.last_pose = None
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Clean up
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows are closed properly

if __name__ == "__main__":
    main()
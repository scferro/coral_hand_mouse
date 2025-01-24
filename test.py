#!/usr/bin/env python3
import cv2
import numpy as np

def get_max_resolution_camera(camera_index=0):
    """
    Open a camera device and attempt to find its maximum supported resolution.
    
    Args:
        camera_index (int): Index of the camera device to open. Defaults to 0 (first camera).
    
    Returns:
        cv2.VideoCapture: Opened camera with maximum resolution set
    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    # List of common resolutions to try, from highest to lowest
    resolution_attempts = [
        # (4096, 3072),   # 12MP
        # (3840, 2160),   # 4K
        # (1920, 1080),   # Full HD
        # (1280, 720),    # HD
        (640, 480)      # VGA
    ]
    
    # Try to set the highest resolution
    for width, height in resolution_attempts:
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Capture a frame to verify resolution
        ret, frame = cap.read()
        
        if ret:
            # Verify the actual captured frame matches our attempt
            actual_height, actual_width = frame.shape[:2]
            print(f"Successfully set resolution to {actual_width}x{actual_height}")
            break
    
    return cap

def main():
    # Open the camera at maximum resolution
    cap = get_max_resolution_camera()
    
    if cap is None:
        print("Failed to open camera.")
        return
    
    try:
        # Continuous frame capture loop
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if frame is successfully captured
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the frame
            cv2.imshow('Camera Feed', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Always release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
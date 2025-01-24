# calibration.py
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional
from raw_image_proc.image_proc import Camera, CameraCalibrationData

# Create a directory to store calibration results
os.makedirs('calibration_results', exist_ok=True)

class CameraCalibrator:
    def __init__(self, checkerboard_size=(8, 6), square_size=0.019, 
                 capture_size=(320, 240), target_frames=50):
        """
        Initialize the camera calibrator.
        
        Args:
            checkerboard_size: Number of internal corners on checkerboard (width, height)
            square_size: Physical size of each checkerboard square in meters
            capture_size: Resolution used for processing
            target_frames: Number of successful calibration frames to collect
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.capture_size = capture_size
        self.target_frames = target_frames
        
        # Create the object points array
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration points
        self.objpoints = []
        self.imgpoints = []
        
    def add_frame(self, img: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process a frame for calibration.
        
        Args:
            img: Input frame to process
            
        Returns:
            Tuple of (success, visualization image)
        """
        # Convert to grayscale if needed
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create color version for visualization
        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            
            # Draw corners
            cv2.drawChessboardCorners(vis_img, self.checkerboard_size, corners, ret)
            
            # Add frame counter
            cv2.putText(vis_img, f"Frames: {len(self.imgpoints)}", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return ret, vis_img
    
    def calibrate(self, img_size: Tuple[int, int]) -> dict:
        """
        Perform camera calibration using collected frames.
        
        Args:
            img_size: Image resolution (width, height)
            
        Returns:
            Dictionary containing calibration parameters
        """
        if not self.objpoints:
            raise ValueError("No calibration data collected")
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None)
        
        # Package results
        params = {
            'resolution': list(img_size),
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'calibration_error': float(ret)
        }
        
        return params
    
    def save_calibration(self, params: dict, filename: str) -> None:
        """Save calibration parameters to file"""
        filepath = os.path.join('calibration_results', filename)
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)

def main():
    print("Initializing calibration...")

    # Set calibration parameters
    target_frames = 250  # Reduced from 500 since we only need one camera
    capture_size = (320, 240)
    
    # Initialize calibrator
    calibrator = CameraCalibrator(
        checkerboard_size=(8, 6),    # For a 9x7 checkerboard
        square_size=0.019,           # 19mm squares
        capture_size=capture_size,   # Processing resolution
        target_frames=target_frames  # Number of calibration frames to capture
    )
    
    print("Opening camera...")
    camera = Camera(target_size=capture_size)
    
    # Verify camera setup
    frame = camera.get_frame()
    if frame is None:
        print("Error: Could not capture frame from camera")
        return
    
    print(f"\nCamera setup verification:")
    print(f"Frame shape: {frame.shape}")
    
    frame_count = 0
    
    print("\nStarting camera calibration:")
    print("1. Hold the checkerboard in front of the camera")
    print("2. Move it slowly to different positions and angles")
    print("3. Try to cover the entire field of view")
    print(f"Need {target_frames} good captures. Press 'q' to quit.\n")
    
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    
    while frame_count < target_frames:
        frame = camera.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        
        success, vis_img = calibrator.add_frame(frame)
        
        if success:
            frame_count += 1
            print(f"\nSuccess! Captured frame {frame_count}/{target_frames}")
        
        if vis_img is not None:
            cv2.imshow('Calibration', vis_img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("\nCalibration cancelled by user")
            break
    
    print("\nCleaning up...")
    camera.cleanup()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    if frame_count > 0:
        print("\nCalculating calibration parameters...")
        params = calibrator.calibrate(capture_size[::-1])  # OpenCV uses (height, width)
        
        calibrator.save_calibration(params, 'camera_calibration.json')
        print("\nCalibration complete! Parameters saved to 'calibration_results/camera_calibration.json'")
    else:
        print("\nNo frames captured. Calibration failed.")

if __name__ == "__main__":
    main()
# main.py
import cv2
import numpy as np
import time
import json
import argparse
from typing import Optional, Tuple, Dict
from image_proc.image_proc import (
    StereoCamera,
    StereoCameraCalibrationData,
    CameraCalibrationData
)
from depth.depth_estimation import DepthEstimator, DepthConfig

class CameraSystem:
    """Main system for stereo camera handling and depth estimation"""
    
    def __init__(self, display: bool = True, calib_file: Optional[str] = None):
        """
        Initialize the stereo camera system with depth estimation.
        
        Args:
            display: Whether to show visualization windows
            calib_file: Path to calibration file
        """
        # Initialize stereo camera with full resolution
        self.camera = StereoCamera(combined_size=(1280, 480))
        
        # Load camera calibration if provided
        if calib_file:
            print('Loading saved camera calibration.')
            try:
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                
                # Create individual camera calibration objects
                left_calib = CameraCalibrationData(
                    camera_matrix=np.array(calib_data['left_camera']['camera_matrix']),
                    dist_coeffs=np.array(calib_data['left_camera']['dist_coeffs']),
                    resolution=tuple(calib_data['resolution']['single'])
                )
                
                right_calib = CameraCalibrationData(
                    camera_matrix=np.array(calib_data['right_camera']['camera_matrix']),
                    dist_coeffs=np.array(calib_data['right_camera']['dist_coeffs']),
                    resolution=tuple(calib_data['resolution']['single'])
                )
                
                # Create stereo calibration object
                stereo_calib = StereoCameraCalibrationData(
                    left_cam=left_calib,
                    right_cam=right_calib,
                    R=np.array(calib_data['stereo']['R']),
                    T=np.array(calib_data['stereo']['T']),
                    E=np.array(calib_data['stereo']['E']),
                    F=np.array(calib_data['stereo']['F']),
                    Q=np.array(calib_data['stereo']['Q'])
                )
                
                self.camera.set_calibration(stereo_calib)
                print("Calibration loaded successfully")
                
            except Exception as e:
                print(f"Warning: Failed to load calibration file: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No calibration file provided. Depth estimation will be unavailable.")
        
        # Initialize depth estimator with conservative parameters
        self.depth_estimator = DepthEstimator(DepthConfig(
            min_depth=0.1,
            max_depth=4.0,  # Maximum depth in meters
            window_size=11,  # Block size for stereo matching
            speckle_size=100,
            speckle_range=1,
            uniqueness_ratio=15
        ))
        
        self.display = display
        self.show_depth = True  # Toggle for depth visualization
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_update_interval = 10.0
    
    def _process_frames(self, left_frame: np.ndarray, 
                    right_frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process stereo frames and compute depth if available.
        
        Args:
            left_frame: Left camera image
            right_frame: Right camera image
            
        Returns:
            Tuple of (stereo_frame, depth_visualization, masked_color)
        """
        # Apply undistortion if calibration is available
        left_undist, right_undist = self.camera.undistort_frames(left_frame, right_frame)
        
        # Create the combined stereo image
        stereo_frame = np.hstack((left_undist, right_undist))
        
        # Initialize visualizations as None
        depth_vis = None
        masked_color = None
        
        # Compute depth if calibration is available and depth visualization is enabled
        if self.camera.calib_data is not None and self.show_depth:
            try:
                _, depth_vis, masked_color = self.depth_estimator.compute_depth(
                    left_undist, right_undist,
                    self.camera.calib_data.Q
                )
            except Exception as e:
                print(f"Depth computation error: {e}")
        
        return stereo_frame, depth_vis, masked_color
    
    def run(self) -> None:
        """Main processing loop with visualization"""
        if self.display:
            cv2.namedWindow('Stereo Feed', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Masked Color', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo Feed', 1280, 480)
            cv2.resizeWindow('Depth Map', 640, 480)
            cv2.resizeWindow('Masked Color', 640, 480) 
        
        while True:
            # Get stereo frames
            left_frame, right_frame = self.camera.get_frames()
            if left_frame is None or right_frame is None:
                print("Failed to capture frames")
                break
            
            # Process frames and compute depth
            stereo_frame, depth_vis, masked_color = self._process_frames(left_frame, right_frame)
            
            # Update and display FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.fps_update_interval:
                fps = self.frame_count / elapsed_time
                print(f"Average FPS: {fps:.1f}")
                self.frame_count = 0
                self.start_time = time.time()
            
            # Display results
            if self.display:
                # Show stereo feed with status
                status = "Calibrated" if self.camera.calib_data is not None else "Uncalibrated"
                cv2.putText(stereo_frame, f"Camera Status: {status}", 
                            (10, stereo_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Stereo Feed', stereo_frame)
                
                # Show depth map and masked color if available
                if depth_vis is not None:
                    cv2.imshow('Depth Map', depth_vis)
                if masked_color is not None:
                    cv2.imshow('Masked Color', masked_color)  # Add this line
                
                # Show depth map if available
                if depth_vis is not None:
                    cv2.imshow('Depth Map', depth_vis)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting camera view...")
                    break
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                    print(f"\nDepth visualization: {'enabled' if self.show_depth else 'disabled'}")
        
        self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.camera.cleanup()
        if self.display:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.waitKey(1)

def main():
    """Main entry point with command line argument handling"""
    parser = argparse.ArgumentParser(
        description='Stereo Camera Application with Depth Estimation')
    
    parser.add_argument('--display', action='store_true',
                      help='Run without displaying video feed')
    parser.add_argument('--calib-file', type=str,
                      default='calibration_results/stereo_calibration.json',
                      help='Path to stereo calibration file')
    
    args = parser.parse_args()
    
    try:
        print("\nInitializing stereo camera system...")
        
        system = CameraSystem(
            display=args.display,
            calib_file=args.calib_file if args.display else None
        )
        
        print("\nStarting camera feed...")
        print("Press 'q' to quit")
        print("Press 'd' to toggle depth visualization")
        
        system.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        print("Exiting program")

if __name__ == "__main__":
    main()
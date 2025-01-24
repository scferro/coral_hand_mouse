# image_proc/image_proc.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CameraCalibrationData:
    """Stores calibration data for a single camera"""
    camera_matrix: np.ndarray  # 3x3 camera intrinsic matrix
    dist_coeffs: np.ndarray    # Distortion coefficients
    resolution: Tuple[int, int]  # Camera resolution (width, height)

class Camera:
    """Single camera implementation optimized for Coral Dev Board"""
    def __init__(self, camera_index: int = 0, target_size: Tuple[int, int] = (320, 240)):
        """
        Initialize camera system with optional resolution scaling.
        
        Args:
            camera_index: Index of the camera device to use
            target_size: Desired output resolution (width, height)
        """
        # Open camera with V4L2 backend for better performance on Linux systems
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        # Store desired output resolution
        self.target_size = target_size
        
        # Initialize calibration data as None - will be set after calibration
        self.calib_data: Optional[CameraCalibrationData] = None
        
        # Set camera properties for desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and process a single frame.
        
        Returns:
            Frame in BGR format, or None if capture failed
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Only resize if necessary (if camera couldn't set exact resolution)
        if frame.shape[:2][::-1] != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        return frame

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply undistortion to a frame using stored calibration parameters.
        
        Args:
            frame: Input frame to undistort
            
        Returns:
            Undistorted frame, or original frame if no calibration data available
        """
        if self.calib_data is None:
            return frame
            
        return cv2.undistort(
            frame, 
            self.calib_data.camera_matrix,
            self.calib_data.dist_coeffs
        )

    def cleanup(self) -> None:
        """Release camera resources"""
        self.cap.release()

    def set_calibration(self, calib_data: CameraCalibrationData) -> None:
        """
        Set camera calibration parameters.
        
        Args:
            calib_data: Calibration data object containing camera parameters
        """
        self.calib_data = calib_data
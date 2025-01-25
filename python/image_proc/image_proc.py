# image_proc/image_proc.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class CameraCalibrationData:
    """Stores calibration data for a single camera"""
    camera_matrix: np.ndarray  # 3x3 camera intrinsic matrix
    dist_coeffs: np.ndarray    # Distortion coefficients
    resolution: Tuple[int, int]  # Camera resolution (width, height)

@dataclass
class StereoCameraCalibrationData:
    """Stores calibration data for stereo camera setup"""
    left_cam: CameraCalibrationData
    right_cam: CameraCalibrationData
    R: np.ndarray  # Rotation matrix between cameras
    T: np.ndarray  # Translation vector between cameras (baseline, 0, 0)
    E: np.ndarray  # Essential matrix
    F: np.ndarray  # Fundamental matrix
    Q: Optional[np.ndarray] = None  # Disparity-to-depth mapping matrix

class StereoCamera:
    """Stereo camera implementation for side-by-side video feed"""
    def __init__(self, camera_index: int = 0, 
                 combined_size: Tuple[int, int] = (1280, 480)):
        """
        Initialize stereo camera system.
        
        Args:
            camera_index: Index of the camera device
            combined_size: Size of the combined stereo feed (width, height)
                         Width should be twice the individual camera width
        """
        if combined_size[0] % 2 != 0:
            raise ValueError("Combined width must be even for stereo split")
        
        # Open camera with V4L2 backend
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        # Store configuration
        self.combined_size = combined_size
        # Individual camera resolution is half the width
        self.single_size = (combined_size[0] // 2, combined_size[1])
        
        self.calib_data: Optional[StereoCameraCalibrationData] = None
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, combined_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, combined_size[1])
        
        # Verify settings were applied
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_width, actual_height) != combined_size:
            print(f"Warning: Camera returned size {actual_width}x{actual_height} "
                  f"instead of requested {combined_size[0]}x{combined_size[1]}")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture and split the side-by-side stereo feed into separate frames.
        
        Returns:
            Tuple of (left_frame, right_frame) in BGR format, or (None, None) if capture failed
        """
        ret, combined_frame = self.cap.read()
        if not ret:
            return None, None
        
        # Ensure frame is the expected size
        if combined_frame.shape[:2][::-1] != self.combined_size:
            combined_frame = cv2.resize(combined_frame, self.combined_size, 
                                     interpolation=cv2.INTER_AREA)
        
        # Split the frame down the middle
        mid_point = self.combined_size[0] // 2
        left_frame = combined_frame[:, :mid_point]
        right_frame = combined_frame[:, mid_point:]
        
        return left_frame, right_frame

    def undistort_frames(self, left_frame: np.ndarray, 
                        right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply undistortion to stereo frames using stored calibration parameters.
        
        Args:
            left_frame: Input left frame
            right_frame: Input right frame
            
        Returns:
            Tuple of undistorted (left_frame, right_frame)
        """
        if self.calib_data is None:
            return left_frame, right_frame
            
        left_undist = cv2.undistort(
            left_frame, 
            self.calib_data.left_cam.camera_matrix,
            self.calib_data.left_cam.dist_coeffs
        )
        
        right_undist = cv2.undistort(
            right_frame, 
            self.calib_data.right_cam.camera_matrix,
            self.calib_data.right_cam.dist_coeffs
        )
        
        return left_undist, right_undist

    def cleanup(self) -> None:
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()

    def set_calibration(self, calib_data: StereoCameraCalibrationData) -> None:
        """Set stereo camera calibration parameters."""
        self.calib_data = calib_data
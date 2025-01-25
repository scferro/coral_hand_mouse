# depth/depth_estimation.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class DepthConfig:
    """Configuration parameters for depth estimation"""
    min_depth: float = 0.1  # Minimum depth in meters
    max_depth: float = 4.0  # Maximum depth in meters
    window_size: int = 11  # Block size for stereo matching
    speckle_size: int = 100
    speckle_range: int = 1
    uniqueness_ratio: int = 9
    display_colormap: int = cv2.COLORMAP_JET
    display_mask: bool = True  # Whether to show masked regions
    mask_color: Tuple[int, int, int] = (128, 0, 128)  # Purple for invalid regions
    masked_alpha: float = 0.5  # Transparency for masked regions in color image

class DepthEstimator:
    """Handles depth estimation and visualization from stereo images"""
    
    def __init__(self, config: Optional[DepthConfig] = None):
        """Initialize the depth estimator with configuration parameters."""
        self.config = config or DepthConfig()
        
        # Create stereo matcher with configured parameters
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Must be divisible by 16
            blockSize=self.config.window_size,
            P1=8 * 3 * self.config.window_size ** 2,
            P2=32 * 3 * self.config.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=self.config.uniqueness_ratio,
            speckleWindowSize=self.config.speckle_size,
            speckleRange=self.config.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Create WLS filter for disparity refinement
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
    
    def compute_depth(self, left_frame: np.ndarray, right_frame: np.ndarray, 
                    Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute depth map and create visualizations with invalid pixels masked black.
        
        Args:
            left_frame: Left camera image
            right_frame: Right camera image
            Q: Perspective transformation matrix from calibration
            
        Returns:
            Tuple of (depth_map, depth_visualization, masked_color_image)
        """
        try:
            # Convert to grayscale for matching while keeping color for visualization
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute disparities for both images
            left_disp = self.stereo.compute(left_gray, right_gray)
            right_disp = self.right_matcher.compute(right_gray, left_gray)
            
            # Convert to float and scale
            left_disp = left_disp.astype(np.float32) / 16.0
            right_disp = right_disp.astype(np.float32) / 16.0
            
            # Apply WLS filter for refinement
            filtered_disp = self.wls_filter.filter(
                left_disp, left_gray, disparity_map_right=right_disp)
            
            # Reproject to 3D
            points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)
            
            # Extract depth (Z component)
            depth_map = points_3d[:, :, 2]
            
            # Create mask for invalid depths
            invalid_mask = np.zeros_like(depth_map, dtype=bool)
            invalid_mask |= (depth_map >= self.config.max_depth)  # Too far
            invalid_mask |= (depth_map <= self.config.min_depth)  # Too close
            invalid_mask |= np.isnan(depth_map)  # Invalid calculations
            invalid_mask |= np.isinf(depth_map)  # Infinite values
            invalid_mask |= (depth_map == 0)  # Zero depth values
            invalid_mask |= (np.mean(left_frame, axis=2) < 10)  # Very dark pixels
            
            # Clean up depth map
            depth_map = np.clip(depth_map, self.config.min_depth, self.config.max_depth)
            
            # Create depth visualization
            norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.applyColorMap(norm_depth, self.config.display_colormap)
            
            # Create fully masked color image (black where invalid)
            masked_color = left_frame.copy()
            masked_color[invalid_mask] = 0  # Set invalid pixels to black
            
            # Make invalid regions black in depth visualization
            depth_vis[invalid_mask] = 0
            
            # Add depth scale bar
            depth_vis = self._add_depth_scale(depth_vis, invalid_mask)
            
            return depth_map, depth_vis, masked_color
            
        except Exception as e:
            print(f"Error in depth computation: {e}")
            return (np.zeros_like(left_frame[:,:,0]), 
                    np.zeros_like(left_frame), 
                    np.zeros_like(left_frame))

    def _add_depth_scale(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add depth scale bar with masking indicator to the visualization."""
        height, width = image.shape[:2]
        scale_width = 30
        margin = 10
        
        # Create scale bar region
        for i in range(height - 2 * margin):
            depth = self.config.max_depth - (i / (height - 2 * margin)) * \
                   (self.config.max_depth - self.config.min_depth)
            value = int(255 * (depth - self.config.min_depth) / 
                       (self.config.max_depth - self.config.min_depth))
            color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), 
                                    self.config.display_colormap)[0, 0]
            cv2.rectangle(image, 
                         (width - scale_width - margin, margin + i),
                         (width - margin, margin + i + 1),
                         color.tolist(), -1)
        
        # Add depth labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        
        # Add depth values
        depths = [self.config.min_depth, 
                 (self.config.min_depth + self.config.max_depth) / 2, 
                 self.config.max_depth]
        
        for depth in depths:
            y_pos = int(margin + (height - 2 * margin) * 
                       (self.config.max_depth - depth) / 
                       (self.config.max_depth - self.config.min_depth))
            cv2.putText(image, f"{depth:.1f}m",
                       (width - scale_width - margin - 45, y_pos),
                       font, font_scale, font_color, font_thickness)
        
        # Add masked region indicator
        if self.config.display_mask:
            masked_percent = (mask.sum() / mask.size) * 100
            cv2.rectangle(image,
                         (width - scale_width - margin, height - margin - 20),
                         (width - margin, height - margin),
                         self.config.mask_color, -1)
            cv2.putText(image, f"Invalid {masked_percent:.1f}%", 
                       (width - scale_width - margin - 70, height - margin - 5),
                       font, font_scale, font_color, font_thickness)
        
        return image
# pose/hand_pose.py
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List
from mediapipe.tasks.python import vision
import scipy.spatial.transform as transform
import time

@dataclass
class PoseConfig:
    """Configuration parameters for hand pose estimation"""
    # Camera parameters (adjust based on your camera)
    focal_length: float = 500.0  # Approximate focal length in pixels
    principal_point: Tuple[float, float] = (320.0, 240.0)  # Image center (half of resolution)
    
    # Hand model parameters
    hand_scale: float = 0.08  # Approximate hand size in meters (8cm default)
    smoothing_factor: float = 0.7  # Temporal smoothing (0 to 1)
    min_confidence: float = 0.7  # Minimum confidence for valid pose
    
    # Debug parameters
    debug: bool = False

@dataclass
class HandPose:
    """Represents the 6-DOF pose of a hand"""
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # 3x3 rotation matrix
    confidence: float
    timestamp: float

class HandPoseEstimator:
    """Estimates 6-DOF hand pose from MediaPipe landmarks"""
    
    # Key hand landmarks used for pose estimation
    LANDMARKS = {
        'WRIST': 0,       # Origin of our coordinate system
        'THUMB_CMC': 1,   # Base of thumb
        'THUMB_MCP': 2,   # First thumb joint
        'INDEX_MCP': 5,   # Base of index finger
        'MIDDLE_MCP': 9,  # Base of middle finger
        'PINKY_MCP': 17,  # Base of pinky finger
        'INDEX_PIP': 6,   # Middle joint of index finger
        'MIDDLE_PIP': 10  # Middle joint of middle finger
    }
    
    def __init__(self, config: Optional[PoseConfig] = None):
        """
        Initialize the pose estimator.
        
        Args:
            config: Configuration parameters for pose estimation
        """
        self.config = config or PoseConfig()
        
        # Create camera matrix from configuration
        self.camera_matrix = np.array([
            [self.config.focal_length, 0, self.config.principal_point[0]],
            [0, self.config.focal_length, self.config.principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize state variables
        self.last_pose: Optional[HandPose] = None
        self.model_points = self._create_hand_model(self.config.hand_scale)
        
        if self.config.debug:
            print("Hand pose estimator initialized with configuration:")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Model points:\n{self.model_points}")
    
    def _create_hand_model(self, scale: float) -> np.ndarray:
        """
        Create an anatomically-aligned 3D hand model.
        
        The model defines a right-handed coordinate system where:
        - Origin is at the wrist
        - X-axis points towards the thumb (red)
        - Y-axis points along the middle finger (green)
        - Z-axis points out of the palm (blue)
        """
        model = np.array([
            [0.0, 0.0, 0.0],        # WRIST (origin)
            [0.5, -0.2, 0.1],       # THUMB_CMC (raised from palm)
            [0.8, 0.0, 0.2],        # THUMB_MCP
            [0.0, 1.0, 0.0],        # INDEX_MCP
            [0.0, 1.1, 0.0],        # MIDDLE_MCP
            [-0.5, 0.9, 0.0],       # PINKY_MCP
            [0.0, 1.4, 0.2],        # INDEX_PIP (raised)
            [0.0, 1.5, 0.2]         # MIDDLE_PIP (raised)
        ], dtype=np.float32) * scale
        
        return model
    
    def estimate_pose(self, detection_result: vision.HandLandmarkerResult,
                     image_size: Tuple[int, int]) -> Optional[HandPose]:
        """
        Estimate 6-DOF hand pose from MediaPipe landmarks.
        
        Args:
            detection_result: MediaPipe hand detection result
            image_size: Tuple of (width, height) of the input image
            
        Returns:
            HandPose object containing position and orientation, or None if estimation fails
        """
        if not detection_result.hand_landmarks:
            return None
        
        try:
            landmarks = detection_result.hand_landmarks[0]
            
            # Calculate initial pose from palm frame
            rotation, translation = self._calculate_palm_frame(landmarks)
            
            # Get 2D points for PnP refinement
            points_2d = np.array([
                [landmarks[idx].x * image_size[0], 
                 landmarks[idx].y * image_size[1]]
                for idx in self.LANDMARKS.values()
            ], dtype=np.float32)
            
            # Refine pose using PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points,
                points_2d,
                self.camera_matrix,
                None,  # No distortion
                cv2.Rodrigues(rotation)[0],  # Initial rotation
                translation.reshape(3, 1),    # Initial translation
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                if self.config.debug:
                    print("PnP refinement failed, using initial estimate")
                rotation_matrix = rotation
                translation_final = translation
                confidence = 0.5
            else:
                # Convert rotation vector to matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                translation_final = translation_vec.flatten()
                
                # Calculate confidence from reprojection error
                projected_points, _ = cv2.projectPoints(
                    self.model_points,
                    rotation_vec,
                    translation_vec,
                    self.camera_matrix,
                    None
                )
                error = np.mean(np.abs(points_2d - projected_points.reshape(-1, 2)))
                confidence = max(0.0, 1.0 - (error / 10.0))
            
            # Create pose object
            pose = HandPose(
                position=translation_final,
                orientation=rotation_matrix,
                confidence=confidence,
                timestamp=time.time()
            )
            
            # Apply temporal smoothing if confidence is high enough
            if confidence >= self.config.min_confidence:
                pose = self._smooth_pose(pose)
            
            return pose
            
        except Exception as e:
            if self.config.debug:
                print(f"Pose estimation error: {str(e)}")
            return None
    
    def _calculate_palm_frame(self, landmarks: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the hand's coordinate frame from landmarks.
        Returns rotation matrix and translation vector.
        """
        # Extract key points
        wrist = np.array([landmarks[self.LANDMARKS['WRIST']].x,
                         landmarks[self.LANDMARKS['WRIST']].y,
                         landmarks[self.LANDMARKS['WRIST']].z])
        
        index_mcp = np.array([landmarks[self.LANDMARKS['INDEX_MCP']].x,
                             landmarks[self.LANDMARKS['INDEX_MCP']].y,
                             landmarks[self.LANDMARKS['INDEX_MCP']].z])
        
        pinky_mcp = np.array([landmarks[self.LANDMARKS['PINKY_MCP']].x,
                             landmarks[self.LANDMARKS['PINKY_MCP']].y,
                             landmarks[self.LANDMARKS['PINKY_MCP']].z])
        
        thumb_cmc = np.array([landmarks[self.LANDMARKS['THUMB_CMC']].x,
                             landmarks[self.LANDMARKS['THUMB_CMC']].y,
                             landmarks[self.LANDMARKS['THUMB_CMC']].z])
        
        # Y-axis: Along fingers (from wrist to middle of index-pinky)
        palm_center = (index_mcp + pinky_mcp) / 2
        y_axis = palm_center - wrist
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # X-axis: Towards thumb, perpendicular to Y
        thumb_dir = thumb_cmc - wrist
        x_axis = thumb_dir - np.dot(thumb_dir, y_axis) * y_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Z-axis: Out of palm using cross product
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Create rotation matrix and translation
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        translation = wrist
        
        return rotation, translation
    
    def _smooth_pose(self, current_pose: HandPose) -> HandPose:
        """Apply temporal smoothing to reduce jitter."""
        if self.last_pose is None:
            self.last_pose = current_pose
            return current_pose
        
        # Smooth position with exponential filter
        smooth_position = (self.config.smoothing_factor * current_pose.position + 
                         (1 - self.config.smoothing_factor) * self.last_pose.position)
        
        # Smooth orientation using quaternion SLERP
        current_quat = transform.Rotation.from_matrix(current_pose.orientation).as_quat()
        last_quat = transform.Rotation.from_matrix(self.last_pose.orientation).as_quat()
        smooth_quat = transform.Slerp(
            [0, 1],
            transform.Rotation.from_quat([last_quat, current_quat])
        )(self.config.smoothing_factor).as_matrix()
        
        smooth_pose = HandPose(
            position=smooth_position,
            orientation=smooth_quat,
            confidence=current_pose.confidence,
            timestamp=current_pose.timestamp
        )
        
        self.last_pose = smooth_pose
        return smooth_pose
    
    def draw_pose(self, frame: np.ndarray, pose: HandPose) -> np.ndarray:
        """Visualize the hand pose with labeled coordinate axes."""
        height, width = frame.shape[:2]
        axis_length = self.config.hand_scale * 200  # Make axes more visible
        
        # Create coordinate axes
        axis_points = np.array([
            [0, 0, 0],                  # Origin
            [axis_length, 0, 0],        # X axis (red, thumb)
            [0, axis_length, 0],        # Y axis (green, fingers)
            [0, 0, axis_length]         # Z axis (blue, palm normal)
        ], dtype=np.float32)
        
        try:
            # Project axes onto image
            rotation_vec = cv2.Rodrigues(pose.orientation)[0]
            projected_points, _ = cv2.projectPoints(
                axis_points,
                rotation_vec,
                pose.position.reshape(3, 1),
                self.camera_matrix,
                None
            )
            
            # Extract projected points
            origin = tuple(map(int, projected_points[0].ravel()))
            point_x = tuple(map(int, projected_points[1].ravel()))
            point_y = tuple(map(int, projected_points[2].ravel()))
            point_z = tuple(map(int, projected_points[3].ravel()))
            
            # Draw coordinate axes
            cv2.line(frame, origin, point_x, (0, 0, 255), 2)  # X: Red
            cv2.line(frame, origin, point_y, (0, 255, 0), 2)  # Y: Green
            cv2.line(frame, origin, point_z, (255, 0, 0), 2)  # Z: Blue
            
            # Draw axis labels
            def draw_axis_label(point, label):
                px, py = point
                size = 0.6 * (1.0 - pose.position[2]/2.0)  # Scale by depth
                size = np.clip(size, 0.4, 0.8)
                cv2.putText(frame, label, (px+10, py+10),
                           cv2.FONT_HERSHEY_SIMPLEX, size,
                           (255, 255, 255), 2)
            
            draw_axis_label(point_x, "X (Thumb)")
            draw_axis_label(point_y, "Y (Fingers)")
            draw_axis_label(point_z, "Z (Palm)")
            
            # Add pose information
            euler_angles = cv2.RQDecomp3x3(pose.orientation)[0]
            text_lines = [
                f"Position (m): ({pose.position[0]:.2f}, {pose.position[1]:.2f}, {pose.position[2]:.2f})",
                f"Rotation (deg): ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f})",
                f"Confidence: {pose.confidence:.2f}"
            ]
            
            # Draw status text
            for i, text in enumerate(text_lines):
                color = (0, 255, 0) if pose.confidence >= self.config.min_confidence else (0, 165, 255)
                cv2.putText(frame, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           color, 2)
                
        except Exception as e:
            if self.config.debug:
                print(f"Visualization error: {str(e)}")
        
        return frame
# mouse/mouse_control.py
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from mediapipe.tasks.python import vision
import time

@dataclass
class MouseConfig:
    """Configuration parameters for mouse control"""
    # Screen mapping parameters
    min_x: float = 0.2  # Add margin to reduce jitter at edges
    max_x: float = 0.8
    min_y: float = 0.2
    max_y: float = 0.8
    
    # Fist click parameters
    fist_threshold: int = 4  # Minimum number of flexed fingers to detect a fist
    
    # Smoothing parameters
    smoothing_factor: float = 0.5  # Mouse movement smoothing (0 to 1)
    click_cooldown: float = 0.05  # Minimum time between clicks in seconds
    
    # Debug parameters
    debug: bool = False
    
    # Handle deprecated parameters for backward compatibility
    def __init__(self, **kwargs):
        # Remove deprecated parameters if present
        kwargs.pop('click_threshold', None)
        kwargs.pop('click_hysteresis', None)
        
        # Set default values for known parameters
        allowed_params = {
            'min_x', 'max_x', 'min_y', 'max_y', 
            'fist_threshold', 
            'smoothing_factor', 
            'click_cooldown', 
            'debug'
        }
        
        # Filter out any unknown parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        
        # Update instance attributes
        for k, v in filtered_kwargs.items():
            setattr(self, k, v)

class HandMouseController:
    """Controls mouse movement and clicks using hand landmarks"""
    
    def __init__(self, config: Optional[MouseConfig] = None):
        """Initialize the hand mouse controller."""
        self.config = config or MouseConfig()
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_click_time: float = 0
        self.is_clicking: bool = False
    
    def update(self, detection_result: vision.HandLandmarkerResult) -> Optional[Dict]:
        """
        Update mouse state based on hand detection.
        
        Args:
            detection_result: MediaPipe hand detection result
            
        Returns:
            Dictionary containing position and click information, or None if no hand detected
        """
        if not detection_result.hand_landmarks:
            return None
        
        # Use the first detected hand for control
        landmarks = detection_result.hand_landmarks[0]
        
        # Get pointer position (using index fingertip)
        raw_x, raw_y = self._get_pointer_position(landmarks)
        screen_x, screen_y = self._map_to_screen(raw_x, raw_y)
        smooth_x, smooth_y = self._smooth_movement(screen_x, screen_y)
        
        # Process clicking using entire hand landmarks
        is_clicking_now = self._process_click_state(landmarks)
        
        return {
            'position': (smooth_x, smooth_y),
            'is_clicking': is_clicking_now,
            'raw_position': (raw_x, raw_y)
        }
    
    def _get_pointer_position(self, landmarks: List) -> Tuple[float, float]:
        """Get normalized coordinates of the pointer (index fingertip)."""
        return landmarks[0].x, landmarks[0].y

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points.
        
        Args:
            p1, p2, p3: Landmark points representing three consecutive points 
                        (e.g., base, middle, and tip of a finger)
        
        Returns:
            Angle in degrees between the vectors
        """
        # Convert landmarks to numpy arrays
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        
        # Prevent division by zero
        if v1_mag == 0 or v2_mag == 0:
            return 180.0
        
        # Calculate cosine and convert to angle
        cos_angle = dot_product / (v1_mag * v2_mag)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Convert to degrees
        return np.degrees(angle)

    def _is_fist_closed(self, landmarks: List) -> bool:
        """
        Determine if the hand is in a closed fist configuration using finger angle analysis.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Boolean indicating if the hand is in a closed fist
        """
        # Define finger landmark sequences 
        # Each finger has landmarks from MCP (base) through PIP (middle) to tip
        finger_sequences = [
            [5, 6, 8],    # Index finger
            [9, 10, 12],  # Middle finger
            [13, 14, 16], # Ring finger
            [17, 18, 20]  # Pinky finger
        ]
        
        # Threshold for considering a finger "bent"
        # Smaller angle indicates more bend
        bend_threshold = 90.0
        
        # Track how many fingers are significantly bent
        flexed_fingers = 0
        
        for finger_seq in finger_sequences:
            # Calculate angle for this finger
            mcp, pip, tip = [landmarks[idx] for idx in finger_seq]
            angle = self._calculate_angle(mcp, pip, tip)
            
            # If angle is small, consider the finger bent
            if angle < bend_threshold:
                flexed_fingers += 1
        
        # Consider it a fist if enough fingers are bent
        return flexed_fingers >= self.config.fist_threshold
    
    def _process_click_state(self, landmarks: List) -> bool:
        """
        Update click state based on fist closure.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Current click state
        """
        current_time = time.time()
        
        # Detect fist closure
        is_fist_closed = self._is_fist_closed(landmarks)
        
        # Use click cooldown to prevent rapid clicking
        if is_fist_closed and not self.is_clicking:
            if current_time - self.last_click_time > self.config.click_cooldown:
                self.is_clicking = True
                self.last_click_time = current_time
        elif not is_fist_closed:
            self.is_clicking = False
        
        return self.is_clicking
    
    def _map_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Map normalized coordinates to screen space with margins."""
        # Clip coordinates to configured boundaries
        x = np.clip(x, self.config.min_x, self.config.max_x)
        y = np.clip(y, self.config.min_y, self.config.max_y)
        
        # Normalize to boundary range
        x = (x - self.config.min_x) / (self.config.max_x - self.config.min_x)
        y = (y - self.config.min_y) / (self.config.max_y - self.config.min_y)
        
        # Convert to screen coordinates (0-100 range)
        screen_x = int(x * 100)
        screen_y = int(y * 100)
        
        return screen_x, screen_y
    
    def _smooth_movement(self, new_x: int, new_y: int) -> Tuple[int, int]:
        """Apply exponential smoothing to reduce jitter."""
        if self.last_position is None:
            self.last_position = (new_x, new_y)
            return new_x, new_y
        
        smooth_x = int(self.config.smoothing_factor * new_x + 
                      (1 - self.config.smoothing_factor) * self.last_position[0])
        smooth_y = int(self.config.smoothing_factor * new_y + 
                      (1 - self.config.smoothing_factor) * self.last_position[1])
        
        self.last_position = (smooth_x, smooth_y)
        return smooth_x, smooth_y
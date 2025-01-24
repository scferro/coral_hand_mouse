# mouse/hand_detection.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class HandDetectionConfig:
    """Configuration parameters for hand detection"""
    model_path: str = 'hand_landmarker.task'
    num_hands: int = 1  # Single hand for better control
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.3
    margin: int = 10
    font_size: float = 0.8
    font_thickness: int = 1
    text_color: Tuple[int, int, int] = (88, 205, 54)  # Light green

class HandDetector:
    """Handles hand detection and landmark visualization using MediaPipe"""
    
    def __init__(self, config: Optional[HandDetectionConfig] = None):
        """
        Initialize the hand detector with optional configuration.
        
        Args:
            config: Configuration parameters for hand detection
        """
        self.config = config or HandDetectionConfig()
        
        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=self.config.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Store the connections for visualization
        self.hand_connections = solutions.hands.HAND_CONNECTIONS
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[vision.HandLandmarkerResult]]:
        """
        Process a single frame to detect hands and visualize landmarks.
        
        Args:
            frame: Input frame in BGR format (OpenCV default)
            
        Returns:
            Tuple of (annotated frame, detection result)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image and detect hands
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        # Draw landmarks on the frame
        annotated_frame = self._draw_landmarks(rgb_frame, detection_result)
        
        # Convert back to BGR for OpenCV display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        return annotated_frame, detection_result
    
    def _draw_landmarks(self, rgb_image: np.ndarray, detection_result: vision.HandLandmarkerResult) -> np.ndarray:
        """
        Draw hand landmarks and handedness labels on the image.
        
        Args:
            rgb_image: Input image in RGB format
            detection_result: MediaPipe hand detection result
            
        Returns:
            Annotated image with landmarks and labels
        """
        annotated_image = np.copy(rgb_image)
        
        if detection_result.hand_landmarks:
            for landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                # Convert landmarks to protobuf format for drawing
                landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in landmarks
                ])
                
                # Draw the landmarks and connections
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    landmarks_proto,
                    self.hand_connections,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )
                
                # Add handedness label
                self._add_handedness_label(
                    annotated_image,
                    landmarks,
                    f"{handedness[0].category_name} Hand"
                )
        
        return annotated_image
    
    def _add_handedness_label(self, image: np.ndarray, landmarks: List, label: str) -> None:
        """Add a text label above the hand landmarks."""
        height, width, _ = image.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - self.config.margin
        
        cv2.putText(
            image, label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            self.config.font_size,
            self.config.text_color,
            self.config.font_thickness,
            cv2.LINE_AA
        )
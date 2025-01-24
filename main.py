# main.py
import cv2
import numpy as np
import time
import json
from typing import Optional
from raw_image_proc.image_proc import Camera, CameraCalibrationData
from mouse.hand_detection import HandDetector, HandDetectionConfig  # Updated import path
from mouse.mouse_control import HandMouseController, MouseConfig

class CameraSystem:
    def __init__(self, display: bool = True, calib_file: Optional[str] = None,
                 model_path: str = 'hand_landmarker.task'):
        """
        Initialize the camera system with hand detection and mouse control.
        
        Args:
            display: Whether to show visualization windows
            calib_file: Path to calibration file. If not provided, uses uncalibrated frames
            model_path: Path to the MediaPipe hand detection model
        """
        # Initialize camera with appropriate resolution
        self.camera = Camera(target_size=(320, 240))
        
        # Set up camera parameters if calibration file is provided
        if calib_file:
            print('Loading saved camera calibration.')
            try:
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                
                self.camera.set_calibration(CameraCalibrationData(
                    camera_matrix=np.array(calib_data['camera_matrix']),
                    dist_coeffs=np.array(calib_data['dist_coeffs']),
                    resolution=tuple(calib_data['resolution'])
                ))
            except Exception as e:
                print(f"Warning: Failed to load calibration file: {str(e)}")
        else:
            print("Warning: No calibration file provided. Images will not be undistorted.")
        
        # Initialize hand detector with custom configuration
        try:
            hand_config = HandDetectionConfig(
                model_path=model_path,
                num_hands=1,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.3
            )
            self.hand_detector = HandDetector(hand_config)
            print("Hand detection initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize hand detection: {str(e)}")
            self.hand_detector = None
        
        # Initialize mouse controller
        try:
            mouse_config = MouseConfig(
                min_x=0.2,
                max_x=0.8,
                min_y=0.2,
                max_y=0.8,
                click_threshold=5.0,
                smoothing_factor=0.5,
                debug=True
            )
            self.mouse_controller = HandMouseController(mouse_config)
            print("Mouse control initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize mouse control: {str(e)}")
            self.mouse_controller = None
        
        self.display = display
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_update_interval = 10.0
    
    def _draw_mouse_visualization(self, frame: np.ndarray, mouse_info: dict) -> np.ndarray:
        """Draw mouse control visualization on the frame."""
        height, width = frame.shape[:2]
        
        # Draw current pointer position
        if mouse_info and 'position' in mouse_info:
            x, y = mouse_info['position']
            pixel_x = int((x / 100) * width)
            pixel_y = int((y / 100) * height)
            
            # Draw crosshair
            color = (0, 255, 0) if mouse_info.get('is_clicking', False) else (0, 255, 255)
            size = 20
            thickness = 2
            cv2.line(frame, (pixel_x - size, pixel_y), (pixel_x + size, pixel_y), color, thickness)
            cv2.line(frame, (pixel_x, pixel_y - size), (pixel_x, pixel_y + size), color, thickness)
            
            # Add status text
            status_text = []
            status_text.append(f"Position: ({x}, {y})")
            if 'thumb_angle' in mouse_info:
                status_text.append(f"Angle: {mouse_info['thumb_angle']:.1f}°")
            status_text.append("CLICKING" if mouse_info.get('is_clicking', False) else "NOT CLICKING")
            
            # Draw status text
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 30 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self) -> None:
        """Main processing loop with hand detection and mouse control"""
        start_time = time.time()
        frame_count = 0
        
        # Create window with a reasonable size
        if self.display:
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Feed', 640, 480)
        
        while True:
            # Get and process frame
            frame = self.camera.get_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Apply undistortion if calibration is available
            processed_frame = self.camera.undistort_frame(frame)
            
            # Ensure we have a 3-channel image for display and processing
            # This is the key fix for the channel error
            if len(processed_frame.shape) == 2:
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = processed_frame.copy()
            
            mouse_info = None
            
            # Perform hand detection and mouse control if available
            if self.hand_detector and self.mouse_controller:
                try:
                    # Detect hand landmarks
                    annotated_frame, detection_result = self.hand_detector.process_frame(display_frame)
                    
                    # Update mouse control
                    if detection_result:
                        mouse_info = self.mouse_controller.update(detection_result)
                    
                    # Add mouse control visualization
                    if mouse_info:
                        display_frame = self._draw_mouse_visualization(annotated_frame, mouse_info)
                    else:
                        display_frame = annotated_frame
                        
                except Exception as e:
                    print(f"Processing error: {str(e)}")
                    # Continue with unprocessed frame if there's an error
                    pass
            
            # Update and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.fps_update_interval:
                fps = frame_count / elapsed_time
                print(f"Average FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()
            
            # Display results if requested
            if self.display:
                # Add status overlay
                status = "Calibrated" if self.camera.calib_data is not None else "Uncalibrated"
                cv2.putText(display_frame, f"Camera Status: {status}", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Camera Feed', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting camera view...")
                    break
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Application with Hand Mouse Control')
    parser.add_argument('--no-display', action='store_true',
                      help='Run without displaying video feed')
    parser.add_argument('--calib-file', type=str,
                      default='calibration_results/camera_calibration.json',
                      help='Path to camera calibration file')
    parser.add_argument('--model-path', type=str,
                      default='hand_landmarker.task',
                      help='Path to MediaPipe hand detection model')
    
    args = parser.parse_args()
    
    try:
        system = CameraSystem(
            display=not args.no_display,
            calib_file=args.calib_file if not args.no_display else None,
            model_path=args.model_path
        )
        system.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        print("Exiting program")

if __name__ == "__main__":
    main()
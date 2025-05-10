import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
from object_detection import draw_segmentation
from fire_detection import detect_fire as detect_fire_impl

# Global class to handle camera and model operations
class ModelCamera:
    def __init__(self, model_path_fire, model_path_box, default_model="fire"):
        # Initialize picamera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (3280, 2464)}))
        self.picam2.start()
        
        # Store model paths
        self.model_path_fire = model_path_fire
        self.model_path_box = model_path_box
        self.current_model = default_model
        self.model = None
        
        # Load default model
        self._load_current_model()
        
    def _load_current_model(self):
        """Load the currently selected model"""
        model_path = self.model_path_fire if self.current_model == "fire" else self.model_path_box
        self.model = YOLO(model_path)
    
    def switch_model(self, model_type):
        """Switch between fire and box detection models"""
        if model_type not in ["fire", "box"]:
            raise ValueError("Model type must be either 'fire' or 'box'")
            
        if model_type != self.current_model:
            self.current_model = model_type
            self._load_current_model()
            print(f"Switched to {model_type} detection model")
    
    def get_frame(self):
        """Capture a frame from the camera"""
        try:
            frame = self.picam2.capture_array()
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame captured")
            
            frame_resized = cv2.resize(frame, (640, 640))
            frame_resized = cv2.convertScaleAbs(frame_resized, alpha=1.2, beta=10)
                
            return frame_resized
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return np.zeros((640, 640, 3), dtype=np.uint8)
    

    


# For backwards compatibility with existing code
# Create a global camera instance
# These paths should be updated to match your actual paths
_fire_model_path = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/fire.pt'
_box_model_path = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/box.pt'
_camera = ModelCamera(_fire_model_path, _box_model_path, default_model="fire")

def get_frame():
    """
    Capture a frame from the Raspberry Pi camera.
    This maintains compatibility with the existing code.
    
    Returns:
        frame (numpy.ndarray): The captured and resized frame.
    """
    return _camera.get_frame()

def detect_fire(frame=None, fire_weights_path=None, confidence_threshold=0.5):
    """
    Wrapper function to maintain compatibility with existing code.
    
    Returns:
        detected: Boolean indicating if fire was detected
        fire_position: Tuple (y, x) of fire position, or None if no fire
        visualization_frame: Frame with fire detection visualization
    """
    return detect_fire_impl(frame, confidence_threshold)

def switch_model(model_type):
    """
    Switch between fire and box detection models.
    
    Args:
        model_type (str): Either "fire" or "box"
    """
    _camera.switch_model(model_type)
#fire_detection.py
import cv2
import numpy as np
from ultralytics import YOLO

def detect_fire(frame, fire_weights_path, confidence_threshold):
    """
    Detect fire in the frame using the fire model.
    
    Returns:
        detected: Boolean indicating if fire was detected
        fire_position: Tuple (y, x) of fire position, or None if no fire
        visualization_frame: Frame with fire detection visualization
    """
    # Load fire detection model
    model = YOLO(fire_weights_path)
    model.conf = confidence_threshold
    
    # Create a copy of the frame for visualization
    visualization_frame = frame.copy()
    
    # Run detection
    results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
    
    fire_position = None
    detected = False
    
    # Process results
    for r in results:
        if len(r.boxes) > 0:
            # Fire detected - get the highest confidence detection
            max_conf_idx = r.boxes.conf.argmax().item()
            confidence = r.boxes.conf[max_conf_idx].item()
            
            # Get coordinates
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[max_conf_idx].cpu().numpy())
            
            # Calculate fire position (center of bounding box)
            fire_y = (y1 + y2) // 2
            fire_x = (x1 + x2) // 2
            
            # Store fire position
            fire_position = (fire_y, fire_x)
            detected = True
            
            # Visualize detection
            cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f'Fire: {confidence:.2f}'
            cv2.putText(visualization_frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return detected, fire_position, visualization_frame
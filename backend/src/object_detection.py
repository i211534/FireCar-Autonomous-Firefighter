#object_detection.py
import cv2
import numpy as np
import heapq
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "ultralytics-main"))
from ultralytics import YOLO

classes = []
COLORS = np.array([])

def load_classes(labels_path):
    """
    Load class labels from file.
    """
    global classes
    if not classes:
        try:
            with open(labels_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(classes)} classes from {labels_path}")
        except FileNotFoundError:
            classes = ['box', 'car']
           # print(f"Warning: Labels file not found at {labels_path}. Using custom class list: {classes}")

def draw_segmentation(img, class_id, confidence, mask, box, enhanced_visualization=False):
    """
    Draw segmentation mask and label for a detected object with only contour lines.
    """
    global classes, COLORS
    if len(classes) <= class_id:
        label = "box"
    else:
        label = str(classes[class_id])
    if COLORS.size == 0 or class_id >= len(COLORS):
        COLORS = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [255, 0, 128]
        ], dtype=np.float32)
    color = COLORS[class_id % len(COLORS)]
    x1, y1, x2, y2 = box
    bright_color = color.astype(np.uint8)
    bright_color_tuple = (int(bright_color[0]), int(bright_color[1]), int(bright_color[2]))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bright_color_tuple, 2)
    label_text = f'{label}: {confidence:.2f}'
    if contours and len(contours) > 0:
        contour = contours[0]
        min_y = y1
        for cnt in contours:
            for point in cnt:
                if point[0][1] < min_y:
                    min_y = point[0][1]
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
        else:
            cx = (x1 + x2) // 2
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = max(0, cx - text_size[0] // 2)
        text_y = max(15, min_y - 5)
        cv2.rectangle(img,
                      (text_x - 2, text_y - text_size[1] - 2),
                      (text_x + text_size[0] + 2, text_y + 2),
                      bright_color_tuple, -1)
        cv2.putText(img, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def segment_objects_and_create_grid(frame, weights_path, labels_path, confidence_threshold, enhanced_visualization=False):
    """
    Segment objects in the frame and create an occupancy grid that preserves object shapes.
    Also detects the car's position.
    
    Returns:
        visualization_frame: Frame with detected objects visualized
        resized_grid: Occupancy grid
        car_position: Tuple (row, col) of the car's position in the grid or None if no car detected
    """

    from func import get_car_position_from_detections    
    global classes, COLORS
    try:
        model = YOLO(weights_path)
        model.conf = confidence_threshold
        model.iou = 0.45
        model.verbose = False
        if not classes:
            try:
                with open(labels_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(classes)} classes from {labels_path}")
            except FileNotFoundError:
                classes = ['box', 'car']
                #print(f"Warning: Labels file not found at {labels_path}. Using default classes: {classes}")
        if COLORS.size == 0:
            COLORS = np.array([
                [0, 255, 0],
                [255, 0, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [255, 128, 0],
                [128, 0, 255],
                [0, 128, 255],
                [255, 0, 128]
            ], dtype=np.float32)
        original_frame = frame.copy()
        results = model.predict(
            source=frame,
            conf=confidence_threshold,
            iou=0.45,
            imgsz=640,
            verbose=False,
            max_det=100,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Extract car position
        Height, Width = frame.shape[:2]
        car_position = get_car_position_from_detections(results, Width, Height, 50, 50)
        
        # Create masks for visualization and grid creation
        precise_shape_mask = np.zeros((Height, Width), dtype=np.uint8)
        car_mask = np.zeros((Height, Width), dtype=np.uint8)
        
        visualization_frame = original_frame.copy()
        detection_count = 0
        
        # Process masks if available
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks.data) > 0:
                for i, mask in enumerate(r.masks.data):
                    class_id = int(r.boxes.cls[i].item()) if i < len(r.boxes.cls) else 0
                    confidence = r.boxes.conf[i].item() if i < len(r.boxes.conf) else 0
                    
                    # Skip if this detection might be a fire (in top portion of frame)
                    if hasattr(r.boxes, 'xyxy'):
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                        # If object is in top right portion and has high confidence, might be fire
                        if y1 < Height * 0.3 and x2 > Width * 0.7:
                            continue  # Skip this detection
                    
                    if confidence >= confidence_threshold:
                        detection_count += 1
                        if hasattr(mask, 'cpu'):
                            mask_np = mask.cpu().numpy()
                        else:
                            mask_np = np.array(mask)
                        mask_cv = cv2.resize(mask_np, (Width, Height))
                        mask_binary = (mask_cv > 0.5).astype(np.uint8) * 255
                        
                        # Check class ID - only add boxes to obstacle mask, not cars
                        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        if class_name.lower() == 'car':
                            car_mask = np.maximum(car_mask, mask_binary)
                        else:
                            precise_shape_mask = np.maximum(precise_shape_mask, mask_binary)
                            
                        if hasattr(r.boxes, 'xyxy'):
                            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                        else:
                            box = r.boxes.data[i]
                            x1, y1, x2, y2 = map(int, box[:4])
                        color = COLORS[class_id % len(COLORS)]
                        bright_color = color.astype(np.uint8)
                        bright_color_tuple = (int(bright_color[0]), int(bright_color[1]), int(bright_color[2]))
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(visualization_frame, contours, -1, bright_color_tuple, 2)
                        if class_id < len(classes):
                            label = str(classes[class_id])
                        else:
                            label = f"class_{class_id}"
                        label_text = f'{label}: {confidence:.2f}'
                        if contours and len(contours) > 0:
                            contour = contours[0]
                            min_y = Height
                            for cnt in contours:
                                for point in cnt:
                                    if point[0][1] < min_y:
                                        min_y = point[0][1]
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                            else:
                                cx = (x1 + x2) // 2
                            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            text_x = max(0, cx - text_size[0] // 2)
                            text_y = max(15, min_y - 5)
                            cv2.rectangle(visualization_frame,
                                          (text_x - 2, text_y - text_size[1] - 2),
                                          (text_x + text_size[0] + 2, text_y + 2),
                                          bright_color_tuple, -1)
                            cv2.putText(visualization_frame, label_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process bounding boxes if no masks
            if not hasattr(r, 'masks') or r.masks is None or len(r.masks.data) == 0:
                for i in range(len(r.boxes.xyxy)):
                    if hasattr(r.boxes, 'xyxy'):
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                    else:
                        box = r.boxes.data[i]
                        x1, y1, x2, y2 = map(int, box[:4])
                    class_id = int(r.boxes.cls[i].item())
                    confidence = r.boxes.conf[i].item()
                    
                    # Skip if this detection might be a fire (in top portion of frame)
                    if y1 < Height * 0.3 and x2 > Width * 0.7:
                        continue  # Skip this detection
                    
                    if confidence >= confidence_threshold:
                        detection_count += 1
                        mask = np.zeros((Height, Width), dtype=np.uint8)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        
                        # Check class ID - only add boxes to obstacle mask, not cars
                        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        if class_name.lower() == 'car':
                            car_mask = np.maximum(car_mask, mask)
                        else:
                            precise_shape_mask = np.maximum(precise_shape_mask, mask)
                            
                        color = COLORS[class_id % len(COLORS)]
                        bright_color = color.astype(np.uint8)
                        bright_color_tuple = (int(bright_color[0]), int(bright_color[1]), int(bright_color[2]))
                        mask_edges = np.zeros((Height, Width), dtype=np.uint8)
                        cv2.rectangle(mask_edges, (x1, y1), (x2, y2), 255, 2)
                        contours, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(visualization_frame, contours, -1, bright_color_tuple, 2)
                        if class_id < len(classes):
                            label = str(classes[class_id])
                        else:
                            label = f"class_{class_id}"
                        label_text = f'{label}: {confidence:.2f}'
                        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_x = max(0, (x1 + x2) // 2 - text_size[0] // 2)
                        text_y = max(15, y1 - 5)
                        cv2.rectangle(visualization_frame,
                                      (text_x - 2, text_y - text_size[1] - 2),
                                      (text_x + text_size[0] + 2, text_y + 2),
                                      bright_color_tuple, -1)
                        cv2.putText(visualization_frame, label_text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Process the obstacle mask and create grid
        binary_mask = (precise_shape_mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        precise_shape_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(precise_shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grid_size = 50
        resized_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        scale_y = grid_size / Height
        scale_x = grid_size / Width
        for contour in contours:
            scaled_contour = []
            for point in contour:
                x, y = point[0]
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_x = max(0, min(scaled_x, grid_size - 1))
                scaled_y = max(0, min(scaled_y, grid_size - 1))
                scaled_contour.append(np.array([[scaled_x, scaled_y]]))
            scaled_contour = np.array(scaled_contour, dtype=np.int32)
            if len(scaled_contour) >= 3:
                cv2.drawContours(resized_grid, [scaled_contour], -1, 1, -1)
            else:
                for point in scaled_contour:
                    x, y = point[0]
                    resized_grid[y, x] = 1
        kernel = np.ones((2, 2), np.uint8)
        resized_grid = cv2.dilate(resized_grid, kernel, iterations=1)
        resized_grid = (resized_grid > 0).astype(int)
        
        # If we have a car position from explicit detection, use it
        # Otherwise, try to determine it from the car mask
        if car_position is None and np.any(car_mask > 0):
            car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if car_contours:
                largest_contour = max(car_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    scaled_x = int(cx * scale_x)
                    scaled_y = int(cy * scale_y)
                    scaled_x = max(0, min(scaled_x, grid_size - 1))
                    scaled_y = max(0, min(scaled_y, grid_size - 1))
                    car_position = (scaled_y, scaled_x)
                    
        # Return the car position along with other results
        return visualization_frame, resized_grid, car_position
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in object detection: {str(e)}")
        cv2.putText(frame, f"Detection Error: {str(e)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, np.zeros((50, 50), dtype=int), None
    
def create_occupancy_grid_from_segmentation(segmentation_mask, height, width):
    """
    Create a 2D numpy array as an occupancy grid that preserves the exact shape of segmented objects.
    
    Args:
        segmentation_mask: Binary mask where 1 indicates occupied space
        height: Grid height
        width: Grid width
    
    Returns:
        2D numpy array representing the occupancy grid
    """
    binary_mask = (segmentation_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occupancy_grid = np.zeros((height, width), dtype=np.uint8)
    scale_y = height / segmentation_mask.shape[0]
    scale_x = width / segmentation_mask.shape[1]
    for contour in contours:
        scaled_contour = []
        for point in contour:
            x, y = point[0]
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_x = max(0, min(scaled_x, width - 1))
            scaled_y = max(0, min(scaled_y, height - 1))
            scaled_contour.append(np.array([[scaled_x, scaled_y]]))
        scaled_contour = np.array(scaled_contour, dtype=np.int32)
        cv2.drawContours(occupancy_grid, [scaled_contour], -1, 1, -1)
    kernel = np.ones((2, 2), np.uint8)
    occupancy_grid = cv2.dilate(occupancy_grid, kernel, iterations=1)
    occupancy_grid = (occupancy_grid > 0).astype(int)
    return occupancy_grid
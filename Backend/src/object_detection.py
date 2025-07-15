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


def segment_objects_and_create_grid(frame, weights_path, labels_path, confidence_threshold, enhanced_visualization=False, fire_detections=None):
    """
    Segment objects in the frame and create an occupancy grid that preserves object shapes.
    Also detects the car's position.
    """
    
    from func import get_car_position_from_detections    
    global classes, COLORS
    
    # Initialize fire_regions as an empty list if none provided
    if fire_detections is None:
        fire_regions = []
    else:
        fire_regions = fire_detections
        print(f"Received {len(fire_regions)} fire detections to exclude from box detection")
    
    try:
        model = YOLO(weights_path)
        model.conf = confidence_threshold
        model.iou = 0.45
        model.verbose = False
        if not classes:
            load_classes(labels_path)
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
        
        # Pre-filter results to remove detections that overlap with fire regions
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0 and hasattr(r.boxes, 'data') and len(r.boxes.data) > 0:
                # Create lists for filtered detections
                filtered_boxes_data = []
                
                # Process each detection
                for i in range(len(r.boxes.data)):
                    box_data = r.boxes.data[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box_data[:4])
                    confidence = box_data[4]
                    class_id = int(box_data[5])
                    
                    # Check if this detection overlaps with any fire region
                    is_overlapping_fire = False
                    for fire_box in fire_regions:
                        if check_overlap((x1, y1, x2, y2), fire_box, threshold=0.05):
                            is_overlapping_fire = True
                            print(f"Skipping detection (class {class_id}) as it overlaps with fire region")
                            break
                    
                    # Keep only non-overlapping detections
                    if not is_overlapping_fire:
                        filtered_boxes_data.append(r.boxes.data[i])
                
                # Create new tensor from filtered list if any boxes remain
                if filtered_boxes_data:
                    # Instead of modifying properties directly, create a new Boxes object
                    # For YOLO, we can just update the data tensor directly
                    r.boxes.data = torch.stack(filtered_boxes_data)
                    # Don't try to set xyxy, conf, or cls directly - they're computed properties
                else:
                    # No boxes left after filtering, create empty tensor with correct shape
                    r.boxes.data = torch.zeros((0, 6), device=r.boxes.data.device)  # 6 columns: 4 for box, 1 for conf, 1 for cls
                
                # If there are masks, filter them too
                if hasattr(r, 'masks') and r.masks is not None and len(r.masks.data) > 0:
                    # Get indices of kept boxes (based on non-fire overlapping boxes)
                    if len(r.boxes.data) == 0:
                        # No boxes left, so no masks should remain
                        if len(r.masks.data) > 0:
                            mask_shape = r.masks.data.shape[1:]
                            r.masks.data = torch.zeros((0, *mask_shape), device=r.masks.data.device)
                    else:
                        # Some boxes remain, keep corresponding masks
                        # Since we've already filtered boxes, masks indices should match
                        # Just ensure we don't have more masks than boxes
                        if len(r.masks.data) > len(r.boxes.data):
                            r.masks.data = r.masks.data[:len(r.boxes.data)]
        
        # Extract car position
        Height, Width = frame.shape[:2]
        car_position = get_car_position_from_detections(results, Width, Height, 50, 50)
        
        # Continue with the rest of the function as before
        precise_shape_mask = np.zeros((Height, Width), dtype=np.uint8)
        # Create a separate mask for car detection
        car_mask = np.zeros((Height, Width), dtype=np.uint8)
        
        visualization_frame = original_frame.copy()
        detection_count = 0
        
        # Create a mask for fire regions to be excluded from the occupancy grid
        fire_exclusion_mask = np.zeros((Height, Width), dtype=np.uint8)
        for fire_box in fire_regions:
            x1, y1, x2, y2 = fire_box
            cv2.rectangle(fire_exclusion_mask, (x1, y1), (x2, y2), 255, -1)
        
        for r in results:
            # Process masks if available
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks.data) > 0:
                # Make sure we only iterate up to the number of available masks and boxes
                num_masks = len(r.masks.data)
                num_boxes = len(r.boxes.data) if hasattr(r.boxes, 'data') else 0
                num_detections = min(num_masks, num_boxes)
                
                for i in range(num_detections):
                    # Get class and confidence using the boxes.cls and boxes.conf properties
                    if i < len(r.boxes.cls) and i < len(r.boxes.conf):  # Use accessor methods
                        class_id = int(r.boxes.cls[i].item())
                        confidence = r.boxes.conf[i].item()
                        
                        if confidence >= confidence_threshold:
                            detection_count += 1
                            mask = r.masks.data[i]
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
                                # Exclude fire regions from the obstacle mask
                                mask_binary = cv2.bitwise_and(mask_binary, cv2.bitwise_not(fire_exclusion_mask))
                                precise_shape_mask = np.maximum(precise_shape_mask, mask_binary)
                                
                            # Get box coordinates using the boxes.xyxy property
                            if hasattr(r.boxes, 'xyxy') and i < len(r.boxes.xyxy):
                                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                            else:
                                box = r.boxes.data[i].cpu().numpy()
                                x1, y1, x2, y2 = map(int, box[:4])
                            
                            # Draw the detection
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
                                # Draw contours and labels as before
                                # ...
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
            
            # Handle case where we have boxes but no masks
            if hasattr(r, 'boxes') and hasattr(r.boxes, 'data') and len(r.boxes.data) > 0:
                if not hasattr(r, 'masks') or r.masks is None or len(r.masks.data) == 0:
                    for i in range(len(r.boxes.data)):
                        # Use accessor properties instead of trying to set them
                        if hasattr(r.boxes, 'xyxy') and i < len(r.boxes.xyxy):
                            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                        else:
                            box = r.boxes.data[i].cpu().numpy()
                            x1, y1, x2, y2 = map(int, box[:4])
                        
                        if i < len(r.boxes.cls) and i < len(r.boxes.conf):
                            class_id = int(r.boxes.cls[i].item())
                            confidence = r.boxes.conf[i].item()
                            
                            if confidence >= confidence_threshold:
                                detection_count += 1
                                mask = np.zeros((Height, Width), dtype=np.uint8)
                                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                                
                                # Check class ID - only add boxes to obstacle mask, not cars
                                class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                                if class_name.lower() == 'car':
                                    car_mask = np.maximum(car_mask, mask)
                                else:
                                    # Exclude fire regions from the obstacle mask
                                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(fire_exclusion_mask))
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
        
        # Ensure fire regions are NOT included in the occupancy grid
        precise_shape_mask = cv2.bitwise_and(precise_shape_mask, cv2.bitwise_not(fire_exclusion_mask))
        
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
        
        # Create a visualization of fire regions on the grid for debugging
        if fire_regions:
            # Convert fire regions to grid coordinates
            fire_grid_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
            for fire_box in fire_regions:
                x1, y1, x2, y2 = fire_box
                grid_x1 = max(0, min(int(x1 * scale_x), grid_size - 1))
                grid_y1 = max(0, min(int(y1 * scale_y), grid_size - 1))
                grid_x2 = max(0, min(int(x2 * scale_x), grid_size - 1))
                grid_y2 = max(0, min(int(y2 * scale_y), grid_size - 1))
                cv2.rectangle(fire_grid_mask, (grid_x1, grid_y1), (grid_x2, grid_y2), 1, -1)
                
            # Ensure fire regions are marked as free space (0) in the grid
            fire_grid_indices = np.where(fire_grid_mask == 1)
            if fire_grid_indices[0].size > 0:
                resized_grid[fire_grid_indices] = 0
                print(f"Cleared {fire_grid_indices[0].size} grid cells for fire regions")
                
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
                    
        print(f"Total detections after filtering: {detection_count}")
        if car_position:
            print(f"Car position: {car_position}")
            
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

# Add this function to object_detection.py
def check_overlap(box1, box2, threshold=0.05):
    """
    Check if two bounding boxes overlap significantly.
    
    Parameters:
        box1: Tuple (x1, y1, x2, y2) of first box
        box2: Tuple (x1, y1, x2, y2) of second box
        threshold: IoU threshold for considering significant overlap
    
    Returns:
        Boolean indicating if boxes overlap significantly
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes actually overlap
    if x2 < x1 or y2 < y1:
        # No direct overlap, but check if they're very close
        # Calculate minimum distance between boxes edges
        h_dist = min(abs(box1[0] - box2[2]), abs(box1[2] - box2[0]))
        v_dist = min(abs(box1[1] - box2[3]), abs(box1[3] - box2[1]))
        
        # If boxes are within 30 pixels of each other, consider them potentially the same object
        proximity_threshold = 30
        return h_dist < proximity_threshold and v_dist < proximity_threshold
    
    # Boxes overlap - calculate IoU
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU (Intersection over Union)
    iou = intersection / float(area1 + area2 - intersection)
    
    # Also consider if one box is mostly contained in the other
    containment_ratio1 = intersection / area1
    containment_ratio2 = intersection / area2
    
    return iou > threshold or containment_ratio1 > 0.5 or containment_ratio2 > 0.5
#func.py
import os
from datetime import datetime
import cv2
from object_detection import segment_objects_and_create_grid
import time

def save_debug_frame(frame, command, position, waypoint_info=None, folder="debug_frames"):
    if frame is None or frame.size == 0:
        return
    os.makedirs(folder, exist_ok=True)
    debug_frame = frame.copy()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cmd_text = f"Command: {command}"
    pos_text = f"Position: {position}"
    height, width = debug_frame.shape[:2]
    text_scale = max(0.5, min(width, height) / 1000)
    thickness = max(1, int(min(width, height) / 500))
    cv2.putText(debug_frame, cmd_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness)
    cv2.putText(debug_frame, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness)
    if waypoint_info:
        wp_text = f"Target: {waypoint_info}"
        cv2.putText(debug_frame, wp_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness)
    cv2.putText(debug_frame, timestamp, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness)
    filename = f"{folder}/cmd_{command}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
    cv2.imwrite(filename, debug_frame)

def get_car_position_from_detections(results, frame_width, frame_height, grid_rows=50, grid_cols=50):
    car_class_id = 1
    if not results:
        return None
    car_position = None
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for i in range(len(r.boxes.cls)):
                class_id = int(r.boxes.cls[i].item())
                confidence = r.boxes.conf[i].item()
                if class_id == car_class_id and confidence >= 0.6:
                    if hasattr(r.boxes, 'xyxy'):
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                    else:
                        box = r.boxes.data[i]
                        x1, y1, x2, y2 = map(int, box[:4])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    grid_row = int(center_y * grid_rows / frame_height)
                    grid_col = int(center_x * grid_cols / frame_width)
                    grid_row = max(0, min(grid_row, grid_rows - 1))
                    grid_col = max(0, min(grid_col, grid_cols - 1))
                    car_position = (grid_row, grid_col)
                    break
        if car_position:
            break
    return car_position

def path_following_thread():
    while True:
        try:
            if path_follower.is_following_path:
                path_follower.follow_path_step()
            time.sleep(0.2)
        except Exception as e:
            print(f"Error in path following thread: {e}")
            time.sleep(1)
def update_position_from_camera_with_frame(self, frame):
    try:
        with self.position_lock:
            if frame is None or frame.size == 0:
                print("Invalid frame provided")
                self.consecutive_position_failures += 1
                return False
                
            _, _, car_position = segment_objects_and_create_grid(frame, self.weights_path, self.labels_path, self.confidence_threshold)
            if car_position:
                print(f"Car detected at grid position: {car_position}")
                self.actual_position = car_position
                self.consecutive_position_failures = 0
                self.last_position_update = time.time()
                return True
            else:
                print("Car not detected in frame")
                self.consecutive_position_failures += 1
                return False
    except Exception as e:
        print(f"Error updating position from camera: {e}")
        self.consecutive_position_failures += 1
        return False
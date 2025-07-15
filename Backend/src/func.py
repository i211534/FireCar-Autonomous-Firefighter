

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


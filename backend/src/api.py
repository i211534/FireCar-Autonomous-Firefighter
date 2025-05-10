#api.py
from flask import Flask, jsonify, send_file, request, Response
import time
import os
from flask_cors import CORS
import cv2
import json
import numpy as np
from object_detection import segment_objects_and_create_grid, create_occupancy_grid_from_segmentation
from path import a_star_car_pathfinding
from camera import get_frame
from fire_detection import detect_fire
app = Flask(__name__)
CORS(app)
fire_x, fire_y = None, None
weights_path = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/merge.pt'
WEIGHTS_PATH = weights_path
fire_detected = False
fire_position = None
FIRE_WEIGHTS_PATH = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/fire.pt'
LABELS_PATH = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/coco2.names'
cap = None
goal_x = 0
goal_y = 0
GRID_ROWS = 20
GRID_COLS = 20
THRESHOLDS = [
    ("low_light", 0.55),
    ("normal", 0.75),
    ("high_contrast", 0.85),
    ("noisy", 0.60),
    ("outdoor", 0.70)
]
current_threshold_idx = 1
CONFIDENCE_THRESHOLD = THRESHOLDS[current_threshold_idx][1]
FIRE_THRESHOLDS = [
    ("low_light", 0.45),
    ("normal", 0.65),
    ("high_contrast", 0.75),
    ("smoky", 0.40),
    ("outdoor", 0.60)
]
current_fire_threshold_idx = 1
FIRE_CONFIDENCE_THRESHOLD = FIRE_THRESHOLDS[current_fire_threshold_idx][1]
@app.route('/thresholds', methods=['GET'])
def get_thresholds():
    return jsonify({
        'object_detection': {
            'current': THRESHOLDS[current_threshold_idx][0],
            'current_value': THRESHOLDS[current_threshold_idx][1],
            'available': [{'name': name, 'value': value} for name, value in THRESHOLDS]
        },
        'fire_detection': {
            'current': FIRE_THRESHOLDS[current_fire_threshold_idx][0],
            'current_value': FIRE_THRESHOLDS[current_fire_threshold_idx][1],
            'available': [{'name': name, 'value': value} for name, value in FIRE_THRESHOLDS]
        }
    })
@app.route('/thresholds/set/<string:name>', methods=['POST'])
def set_threshold(name):
    global current_threshold_idx, CONFIDENCE_THRESHOLD
    for idx, (threshold_name, _) in enumerate(THRESHOLDS):
        if threshold_name.lower() == name.lower():
            current_threshold_idx = idx
            CONFIDENCE_THRESHOLD = THRESHOLDS[current_threshold_idx][1]
            return jsonify({
                'status': 'success',
                'message': f'Threshold set to {threshold_name} ({CONFIDENCE_THRESHOLD})'
            })
    return jsonify({
        'status': 'error',
        'message': f'Threshold "{name}" not found. Available options: {[name for name, _ in THRESHOLDS]}'
    }), 400
@app.route('/thresholds/set_value', methods=['POST'])
def set_threshold_value():
    global CONFIDENCE_THRESHOLD, current_threshold_idx
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold', 0.75))
        if 0.0 <= new_threshold <= 1.0:
            CONFIDENCE_THRESHOLD = new_threshold
            current_threshold_idx = -1
            return jsonify({
                'status': 'success',
                'message': f'Custom threshold set to {CONFIDENCE_THRESHOLD}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Threshold must be between 0.0 and 1.0'
            }), 400
    except (ValueError, TypeError) as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid threshold value: {str(e)}'
        }), 400
@app.route('/fire_thresholds/set/<string:name>', methods=['POST'])
def set_fire_threshold(name):
    global current_fire_threshold_idx, FIRE_CONFIDENCE_THRESHOLD
    for idx, (threshold_name, _) in enumerate(FIRE_THRESHOLDS):
        if threshold_name.lower() == name.lower():
            current_fire_threshold_idx = idx
            FIRE_CONFIDENCE_THRESHOLD = FIRE_THRESHOLDS[current_fire_threshold_idx][1]
            return jsonify({
                'status': 'success',
                'message': f'Fire detection threshold set to {threshold_name} ({FIRE_CONFIDENCE_THRESHOLD})'
            })
    return jsonify({
        'status': 'error',
        'message': f'Threshold "{name}" not found. Available options: {[name for name, _ in FIRE_THRESHOLDS]}'
    }), 400
@app.route('/detect_fire', methods=['GET'])
def detect_fire_api():
    global fire_detected, fire_position, goal_x, goal_y
    threshold = request.args.get('threshold', None)
    temp_threshold = FIRE_CONFIDENCE_THRESHOLD
    if threshold is not None:
        try:
            temp_threshold = float(threshold)
            if not (0.0 <= temp_threshold <= 1.0):
                temp_threshold = FIRE_CONFIDENCE_THRESHOLD
        except ValueError:
            pass
    try:
        frame = get_frame()
        if frame is None or frame.size == 0:
            error_frame = np.zeros((2464, 3280, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error: No valid frame", (200, 1232), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            output_dir = os.path.join(os.path.dirname(__file__), '../data/')
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, 'fire_detection.jpg')
            cv2.imwrite(output_image_path, error_frame)
            return send_file(output_image_path, mimetype='image/jpeg')
        detected, position, visualization_frame = detect_fire(frame, FIRE_WEIGHTS_PATH, confidence_threshold=temp_threshold)
        fire_detected = detected
        fire_position = position
        if fire_detected and fire_position is not None:
            grid_height, grid_width = 50, 50
            frame_height, frame_width = frame.shape[:2]
            grid_y = int(fire_position[0] * grid_height / frame_height)
            grid_x = int(fire_position[1] * grid_width / frame_width)
            goal_y = grid_y
            goal_x = grid_x
        visualization_frame = cv2.resize(visualization_frame, (3280, 2464))
        output_dir = os.path.join(os.path.dirname(__file__), '../data/')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, 'fire_detection.jpg')
        cv2.imwrite(output_image_path, visualization_frame)
        return jsonify({
            'fire_detected': fire_detected,
            'fire_position': fire_position,
            'goal': {"x": goal_x, "y": goal_y} if fire_detected else None,
            'threshold_used': temp_threshold,
            'environment': FIRE_THRESHOLDS[current_fire_threshold_idx][0] if current_fire_threshold_idx >= 0 else 'custom'
        })
    except Exception as e:
        error_frame = np.zeros((2464, 3280, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Fire Detection Error: {str(e)}", (200, 1232), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        output_dir = os.path.join(os.path.dirname(__file__), '../data/')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, 'fire_detection.jpg')
        cv2.imwrite(output_image_path, error_frame)
        return jsonify({'error': str(e), 'fire_detected': False})
@app.route('/detect_objects', methods=['GET'])
def detect_objects_api():
    timestamp = request.args.get('t', int(time.time()))
    threshold = request.args.get('threshold', None)
    temp_threshold = CONFIDENCE_THRESHOLD
    if threshold is not None:
        try:
            temp_threshold = float(threshold)
            if not (0.0 <= temp_threshold <= 1.0):
                temp_threshold = CONFIDENCE_THRESHOLD
        except ValueError:
            pass
    try:
        frame = get_frame()
        if frame is None or frame.size == 0:
            error_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error: No valid frame", (50, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_dir = os.path.join(os.path.dirname(__file__), '../data/')
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(os.path.dirname(__file__), '../data/segmented_live.jpg')
            cv2.imwrite(output_image_path, error_frame)
            return send_file(output_image_path, mimetype='image/jpeg')
        segmented_frame, _ ,_= segment_objects_and_create_grid(
            frame, 
            WEIGHTS_PATH, 
            LABELS_PATH, 
            temp_threshold,
            enhanced_visualization=True
        )
        segmented_frame = cv2.resize(segmented_frame, (1280, 960))
        output_dir = os.path.join(os.path.dirname(__file__), '../data/')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(os.path.dirname(__file__), f'../data/segmented_live_{timestamp}.jpg')
        cv2.imwrite(output_image_path, segmented_frame)
        response = send_file(output_image_path, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        error_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Detection Error: {str(e)}", (50, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_dir = os.path.join(os.path.dirname(__file__), '../data/')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(os.path.dirname(__file__), f'../data/segmented_live_{timestamp}.jpg')
        cv2.imwrite(output_image_path, error_frame)
        response = send_file(output_image_path, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
@app.route('/set_goal', methods=['POST'])
def set_goal():
    global goal_x, goal_y
    data = request.get_json()
    try:
        goal_x = int(data.get('goal_x', 0))
        goal_y = int(data.get('goal_y', 0))
        return jsonify({"message": "Goal coordinates updated successfully.", "goal_x": goal_x, "goal_y": goal_y}), 200
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
def process_navigation_sequence():
    fire_result = detect_fire_api()
    fire_data = json.loads(fire_result.data)
    if fire_data.get('fire_detected', False):
        return get_occupancy_grid()
    else:
        return jsonify({
            'status': 'No fire detected',
            'fire_detected': False,
            'occupancy_grid': [],
            'path': []
        })
@app.route('/navigate_to_fire', methods=['GET'])
def navigate_to_fire():
    global goal_x, goal_y, fire_detected, fire_position
    fire_result = detect_fire_api()
    fire_data = json.loads(fire_result.data)
    if not fire_data.get('fire_detected', False):
        return jsonify({
            'status': 'No fire detected',
            'fire_detected': False,
            'occupancy_grid': [],
            'path': []
        })
    frame = get_frame()
    empty_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    if frame is None or frame.size == 0:
        return jsonify({
            'occupancy_grid': empty_grid.tolist(),
            'path': [],
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': None,
            'status': 'Frame read failure',
            'fire_detected': True
        })
    try:
        segmented_frame, occupancy_grid, car_position = segment_objects_and_create_grid(
            frame, 
            WEIGHTS_PATH, 
            LABELS_PATH, 
            CONFIDENCE_THRESHOLD,
            enhanced_visualization=True
        )
        if occupancy_grid is None or occupancy_grid.size == 0:
            occupancy_grid = empty_grid
            status = 'No objects detected'
        else:
            status = 'Success'
        if car_position:
            start = car_position
        else:
            start = (49, 26)
        goal = (goal_y, goal_x)
        goal = (min(goal[0], occupancy_grid.shape[0]-1), min(goal[1], occupancy_grid.shape[1]-1))
        if occupancy_grid[start[0], start[1]] == 1:
            new_start = find_nearest_free_cell(occupancy_grid, start)
            if new_start:
                start = new_start
        if occupancy_grid[goal[0], goal[1]] == 1:
            new_goal = find_nearest_free_cell(occupancy_grid, goal)
            if new_goal:
                goal = new_goal
        path = []
        try:
            path = a_star_car_pathfinding(occupancy_grid, start, goal)
            if path:
                status += f" | Path found with {len(path)} steps."
                height, width = occupancy_grid.shape
                path_frame = visualize_path(segmented_frame.copy(), path, height, width)
                output_dir = os.path.join(os.path.dirname(__file__), '../data/')
                os.makedirs(output_dir, exist_ok=True)
                output_path_image = os.path.join(os.path.dirname(__file__), '../data/path_visualization.jpg')
                cv2.imwrite(output_path_image, path_frame)
                
        except Exception as path_error:
            path = []
            status = f'Pathfinding failed: {str(path_error)}'
        formatted_path = [[int(y), int(x)] for y, x in path] if path else []
        return jsonify({
            'occupancy_grid': occupancy_grid.tolist(),
            'path': formatted_path,
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': {"y": car_position[0], "x": car_position[1]} if car_position else None,
            'status': status,
            'fire_detected': True,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'occupancy_grid': empty_grid.tolist(),
            'path': [],
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': None,
            'status': f'Error: {str(e)}',
            'fire_detected': True
        })
@app.route('/get_occupancy_grid', methods=['GET'])
def get_occupancy_grid():
    global goal_x, goal_y, fire_detected, fire_position
    threshold = request.args.get('threshold', None)
    temp_threshold = CONFIDENCE_THRESHOLD
    if threshold is not None:
        try:
            temp_threshold = float(threshold)
            if not (0.0 <= temp_threshold <= 1.0):
                temp_threshold = CONFIDENCE_THRESHOLD
        except ValueError:
            pass
    frame = get_frame()
    empty_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    if frame is None or frame.size == 0:
        return jsonify({
            'occupancy_grid': empty_grid.tolist(),
            'path': [],
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': None,
            'status': 'Frame read failure',
            'timestamp': time.time(),
            'threshold_used': temp_threshold
        })
    try:
        segmented_frame, occupancy_grid, car_position = segment_objects_and_create_grid(
            frame, 
            WEIGHTS_PATH, 
            LABELS_PATH, 
            temp_threshold,
            enhanced_visualization=True
        )
        segmented_frame = cv2.resize(segmented_frame, (3280, 2464))
        if occupancy_grid is None or occupancy_grid.size == 0:
            occupancy_grid = empty_grid
            status = 'No objects detected'
        else:
            status = 'Success'
        if car_position:
            start = car_position
        else:
            start = (49, 26)
        goal = (goal_y, goal_x)
        goal = (min(goal[0], occupancy_grid.shape[0]-1), min(goal[1], occupancy_grid.shape[1]-1))
        if occupancy_grid[start[0], start[1]] == 1:
            new_start = find_nearest_free_cell(occupancy_grid, start)
            if new_start:
                start = new_start
        if occupancy_grid[goal[0], goal[1]] == 1:
            new_goal = find_nearest_free_cell(occupancy_grid, goal)
            if new_goal:
                goal = new_goal
        path = []
        try:
            path = a_star_car_pathfinding(occupancy_grid, start, goal)
            if not path:
                status += " | No valid path found."
            else:
                status += f" | Path found with {len(path)} steps."
                height, width = occupancy_grid.shape
                path_frame = visualize_path(segmented_frame.copy(), path, height, width)
                path_frame = cv2.resize(path_frame, (3280, 2464))
                output_dir = os.path.join(os.path.dirname(__file__), '../data/')
                os.makedirs(output_dir, exist_ok=True)
                output_path_image = os.path.join(os.path.dirname(__file__), '../data/path_visualization.jpg')
                cv2.imwrite(output_path_image, path_frame)
        except Exception as path_error:
            path = []
            status = f'Pathfinding failed: {str(path_error)}'
        formatted_path = [[int(y), int(x)] for y, x in path] if path else []     
        return jsonify({
            'occupancy_grid': occupancy_grid.tolist(),
            'path': formatted_path,
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': {"y": car_position[0], "x": car_position[1]} if car_position else None,
            'status': status,
            'timestamp': time.time(),
            'threshold_used': temp_threshold,
            'environment': THRESHOLDS[current_threshold_idx][0] if current_threshold_idx >= 0 else 'custom'
        })
    except Exception as e:
        return jsonify({
            'occupancy_grid': empty_grid.tolist(),
            'path': [],
            'goal': {"x": goal_x, "y": goal_y},
            'car_position': None,
            'status': f'Error: {str(e)}',
            'threshold_used': temp_threshold
        })
def find_nearest_free_cell(grid, point):
    rows, cols = grid.shape
    y, x = point
    for radius in range(1, max(rows, cols)):
        for i in range(max(0, y-radius), min(rows, y+radius+1)):
            for j in range(max(0, x-radius), min(cols, x+radius+1)):
                if i == y-radius or i == y+radius or j == x-radius or j == x+radius:
                    if grid[i, j] == 0:
                        return (i, j)
    return None
def visualize_path(frame, path, grid_height, grid_width):
    height, width = frame.shape[:2]
    scale_y = height / grid_height
    scale_x = width / grid_width
    points = []
    for y, x in path:
        px = int(x * scale_x)
        py = int(y * scale_y)
        points.append((px, py))
    line_thickness = max(3, int(width/640))
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 0), line_thickness)
    circle_radius = max(10, int(width/320))
    text_scale = max(1.0, width/1640)
    text_thickness = max(2, int(width/820))
    if points:
        cv2.circle(frame, points[0], circle_radius, (0, 255, 0), -1)
        cv2.circle(frame, points[-1], circle_radius, (0, 0, 255), -1)
        cv2.putText(frame, "START", (points[0][0] + circle_radius + 5, points[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness)
        cv2.putText(frame, "GOAL", (points[-1][0] + circle_radius + 5, points[-1][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness)
    return frame
@app.route('/api/car/start_following', methods=['POST'])
def start_following():
    global latest_results
    latest_results['is_following_path'] = True
    return jsonify({"status": "success", "message": "Car started following path"})
@app.route('/api/car/stop_following', methods=['POST'])
def stop_following():
    global latest_results
    latest_results['is_following_path'] = False
    return jsonify({"status": "success", "message": "Car stopped following path"})
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
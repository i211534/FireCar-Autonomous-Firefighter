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
from main import visualize_path
from fire_detection import detect_fire
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
app = Flask(__name__)
CORS(app)

# 1?? Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://fire_db_owner:npg_1rhAszIRlj9Z@ep-young-band-a1x28jd0-pooler.ap-southeast-1.aws.neon.tech/fire_db?sslmode=require'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class FireLog(db.Model):
    __tablename__ = 'fire_logs'

    id                = db.Column(db.Integer, primary_key=True)
    timestamp         = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    location          = db.Column(db.String(100), nullable=False)
    severity          = db.Column(db.String(50), nullable=False)
    suppression_action= db.Column(db.String(100), nullable=False)
    resolved          = db.Column(db.Boolean, default=False, nullable=False)

fire_x, fire_y = None, None
weights_path = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/merge.pt'
WEIGHTS_PATH = weights_path
fire_detected = False
fire_position = None
FIRE_WEIGHTS_PATH = r'/home/awaiz/Music/best/backend - Copy - Copy - Copy/models/firereal.pt'
LABELS_PATH = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/coco.names'
cap = None
goal_x = 0
goal_y = 0
GRID_ROWS = 50
GRID_COLS = 50
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
current_fire_threshold_idx = 3
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
    

@app.route('/raw_frame', methods=['GET'])
def raw_frame():
    frame = get_frame()              # grab the latest camera frame
    if frame is None:
        return jsonify({"error":"no frame"}), 500

    # JPEG-encode it in memory
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"error":"encoding failed"}), 500

    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/firelogs', methods=['GET'])
def get_fire_logs():
    logs = FireLog.query.order_by(FireLog.timestamp.desc()).all()
    return jsonify([{
        'id': log.id,
        'timestamp': log.timestamp.isoformat(),
        'location': log.location,
        'severity': log.severity,
        'suppression_action': log.suppression_action,
        'resolved': log.resolved
    } for log in logs]), 200

@app.route('/firelogs', methods=['POST'])
def add_fire_log():
    data = request.get_json()
    log = FireLog(
        location=data['location'],
        severity=data['severity'],
        suppression_action=data['suppression_action'],
        resolved=data.get('resolved', False)
    )
    db.session.add(log)
    db.session.commit()
    return jsonify({'id': log.id, 'message': 'Fire log created'}), 201

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
        
        # Fix the function call to match the signature in main.py
        detected, position, visualization_frame, fire_bbox = detect_fire(frame, FIRE_WEIGHTS_PATH, confidence_threshold=temp_threshold)
        
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
        # Detect fire first to get bounding boxes
        detected, position, _, fire_bbox = detect_fire(frame, FIRE_WEIGHTS_PATH, confidence_threshold=FIRE_CONFIDENCE_THRESHOLD)
        
        # Initialize fire_bboxes as empty list
        fire_bboxes = []
        if detected and position is not None and fire_bbox is not None:
            fire_bboxes.append(fire_bbox)
            
        # Now include fire_bboxes in segment_objects_and_create_grid call
        segmented_frame, occupancy_grid, car_position = segment_objects_and_create_grid(
            frame, 
            WEIGHTS_PATH, 
            LABELS_PATH, 
            CONFIDENCE_THRESHOLD,
            enhanced_visualization=True,
            fire_detections=fire_bboxes  # Add this parameter
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
        # First detect fires to get bounding boxes
        detected_fire, fire_pos, _, fire_bbox = detect_fire(frame, FIRE_WEIGHTS_PATH, confidence_threshold=FIRE_CONFIDENCE_THRESHOLD)
        
        # Initialize fire_bboxes
        fire_bboxes = []
        if detected_fire and fire_pos is not None and fire_bbox is not None:
            fire_bboxes.append(fire_bbox)
            
        # Now segment objects with fire bounding boxes
        segmented_frame, occupancy_grid, car_position = segment_objects_and_create_grid(
            frame, 
            WEIGHTS_PATH, 
            LABELS_PATH, 
            temp_threshold,
            enhanced_visualization=True,
            fire_detections=fire_bboxes  # Pass fire bounding boxes
        )
        
        # Ensure proper grid dimensions - resize if needed
        # This makes sure the grid is consistent with expectations
        if occupancy_grid.shape != (GRID_ROWS, GRID_COLS):
            temp_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
            min_rows = min(occupancy_grid.shape[0], GRID_ROWS)
            min_cols = min(occupancy_grid.shape[1], GRID_COLS)
            temp_grid[:min_rows, :min_cols] = occupancy_grid[:min_rows, :min_cols]
            occupancy_grid = temp_grid
            
        # Debug grid contents
        print(f"Grid dimensions: {occupancy_grid.shape}")
        print(f"Number of obstacles in grid: {np.sum(occupancy_grid)}")
        
        segmented_frame = cv2.resize(segmented_frame, (3280, 2464))
        if occupancy_grid is None or occupancy_grid.size == 0 or np.sum(occupancy_grid) == 0:
            occupancy_grid = empty_grid
            status = 'No objects detected'
        else:
            status = f'Success - detected {np.sum(occupancy_grid)} obstacle cells'
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
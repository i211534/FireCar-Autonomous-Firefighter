import threading
import time
import cv2
import api
import numpy as np
import serial
import os
from datetime import datetime
from camera import get_frame
from object_detection import segment_objects_and_create_grid
from path import a_star_car_pathfinding, find_nearest_free_cell
from fire_detection import detect_fire

temp_threshold = 0.25
save_dir = "command_frames"
os.makedirs(save_dir, exist_ok=True)
FIRE_WEIGHTS_PATH = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/firereal.pt'

def init_serial():
    try:
        ser = serial.Serial('/dev/serial0', 9600)
        time.sleep(2)
        return ser
    except Exception as e:
        print(f"Error initializing serial connection: {e}")
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

# Add these constants at the top of your script
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
DIR_TO_VEC = {
    'NORTH': (-1, 0),
    'EAST': (0, 1),
    'SOUTH': (1, 0),
    'WEST': (0, -1)
}

def update_direction(car_direction, command):
    """
    Update car direction based on command.
    
    Parameters:
    car_direction (str): Current car direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
    command (str): Command to execute ('f', 'l', 'r', 's')
    
    Returns:
    str: New car direction
    """
    idx = DIRECTIONS.index(car_direction)
    if command == 'l':
        return DIRECTIONS[(idx - 1) % 4]  # Left turn
    elif command == 'r':
        return DIRECTIONS[(idx + 1) % 4]  # Right turn
    else:
        return car_direction  # forward or stop

def move_forward(position, direction):
    """
    Move car forward in the current direction.
    
    Parameters:
    position (tuple): Current position (y, x)
    direction (str): Current direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
    
    Returns:
    tuple: New position (y, x)
    """
    dy, dx = DIR_TO_VEC[direction]
    return (position[0] + dy, position[1] + dx)

def process_frame(frame, last_command, ser, current_direction='NORTH', 
                  is_first_command=True, frame_number=0,
                  previous_positions=None, current_path=None, goal_position=None,
                  initial_car_position=None):
    weights_path = r'/home/awaiz/Music/Project/backend - Copy - Copy - Copy/models/box.pt'
    labels_path = 'src/models/coco.names'

    if previous_positions is None:
        previous_positions = []

    threshhold_box = 0.25
    
    # First, run the fire detection separately
    detected_fire, fire_position, fire_confidence = detect_fire(frame, FIRE_WEIGHTS_PATH, temp_threshold)
    
    # Then run the regular object detection (with top-right area skipped)
    segmented_frame, occupancy_grid, car_position = segment_objects_and_create_grid(
        frame, weights_path, labels_path, threshhold_box,
        enhanced_visualization=True
    )

    # Add fire visualization if detected
    if detected_fire and fire_position is not None:
        height, width = frame.shape[:2]
        grid_height, grid_width = 50, 50
        grid_y = int(fire_position[0] * grid_height / height)
        grid_x = int(fire_position[1] * grid_width / width)
        
        # Convert grid position back to pixel coordinates for visualization
        px = int(fire_position[1])
        py = int(fire_position[0])
        
        # Draw fire detection with a distinct color (red)
        fire_radius = max(10, int(width/320))
        cv2.circle(segmented_frame, (px, py), fire_radius, (0, 0, 255), -1)
        
        # Add fire label
        fire_text = f"FIRE: {fire_confidence:.2f}"
        text_size = cv2.getTextSize(fire_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(segmented_frame, 
                    (px - 2, py - text_size[1] - 2),
                    (px + text_size[0] + 2, py + 2),
                    (0, 0, 255), -1)
        cv2.putText(segmented_frame, fire_text, (px, py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update goal position for navigation
        fire_goal_position = (grid_y, grid_x)
        goal_position = fire_goal_position
    
    if car_position is None:
        annotated_frame = save_frame_with_command(segmented_frame, last_command, frame_number)
        return last_command, annotated_frame, occupancy_grid, current_path, current_direction, is_first_command, previous_positions, goal_position, initial_car_position

    # Store initial car position if this is the first frame
    if initial_car_position is None:
        initial_car_position = car_position
        print(f"Initial car position set to {initial_car_position}")

    if len(previous_positions) >= 5:
        previous_positions.pop(0)
    previous_positions.append(car_position)
    
    # Always recalculate path in every frame
    if goal_position:
        print(f"Planning new path from current position {car_position} to goal {goal_position}")
        start = car_position
        path = a_star_car_pathfinding(occupancy_grid, start, goal_position)
        
        if not path:
            print("No path found, trying alternative goal")
            alternative_goal = find_nearest_free_cell(occupancy_grid, goal_position, max_search_radius=20)
            if alternative_goal:
                print(f"Using alternative goal: {alternative_goal}")
                path = a_star_car_pathfinding(occupancy_grid, start, alternative_goal)
                if path:
                    goal_position = alternative_goal
        
        current_path = path
    
    if not current_path:
        print("No path found, stopping the car")
        command = 's'
        send_command(ser, command)
        annotated_frame = save_frame_with_command(segmented_frame, command, frame_number)
        return command, annotated_frame, occupancy_grid, None, current_direction, False, previous_positions, goal_position, initial_car_position
    
    # After calculating the car position and recalculating a new path:
    command, near_goal = determine_command_with_directions(car_position, current_path, current_direction, last_command)
    
    # If near goal and it's a fire, send 'a' command (spray)
    if command == 's' and near_goal and fire_goal_position:
        print("Near fire goal - sending 'a' command to spray.")
        send_command(ser, 'a')
    # NEW: If the command is 's' (stop) then check the front and back occupancy
    elif command == 's' and not near_goal:
        front_free, back_free = check_front_back(occupancy_grid, car_position, current_direction)
        if front_free and back_free:
            print("Both front and back cells are free. Sending forward then backward commands.")
            # Send forward command first
            send_command(ser, 'f')
            
        elif front_free:
            print("Front cell is free. Sending forward command.")
            command = 'f'
        elif back_free:
            print("Back cell is free. Sending backward command.")
            command = 'b'
        else:
            print("Neither front nor back is free. Keeping stop command.")
    
    # Update direction based on (possibly updated) command
    new_direction = update_direction(current_direction, command)
    
    send_command(ser, command)
    
    annotated_frame = save_frame_with_command(segmented_frame, command, frame_number)
    
    if current_path:
        grid_height, grid_width = occupancy_grid.shape
        annotated_frame = visualize_path(annotated_frame, current_path, grid_height, grid_width)
        
        height, width = annotated_frame.shape[:2]
        scale_y = height / grid_height
        scale_x = width / grid_width
        car_pos_px = (int(car_position[1] * scale_x), int(car_position[0] * scale_y))
        
        cv2.circle(annotated_frame, car_pos_px, 8, (255, 0, 0), -1)
        
        # Visualize car direction
        dy, dx = DIR_TO_VEC[current_direction]
        direction_end = (int(car_pos_px[0] + 20 * dx), int(car_pos_px[1] + 20 * dy))
        cv2.line(annotated_frame, car_pos_px, direction_end, (255, 0, 0), 3)
        
        # Add direction text
        cv2.putText(annotated_frame, current_direction, 
                    (car_pos_px[0] + 25, car_pos_px[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return command, annotated_frame, occupancy_grid, current_path, new_direction, False, previous_positions, goal_position, initial_car_position

def send_command(ser, command):
    if ser is None:
        print("Serial connection not available")
        return
        
    valid_commands = {'f': 'forward', 'b': 'backward', 'l': 'left', 'r': 'right', 's': 'stop','a' : 'spray'}
    if command in valid_commands:
        try:
            ser.write(command.encode())
            print(f"Command sent: {command} ({valid_commands[command]})")
            time.sleep(1.5)
            ser.write("s".encode())
            time.sleep(1.5)
        except Exception as e:
            print(f"Error sending command: {e}")
    else:
        print(f"Invalid command: {command}")
 
def save_frame_with_command(frame, command, frame_number):
    annotated_frame = frame.copy()
    
    command_names = {
        'f': 'FORWARD',
        'b': 'BACKWARD',
        'l': 'LEFT',
        'r': 'RIGHT',
        's': 'STOP'
    }
    
    command_name = command_names.get(command, f"UNKNOWN({command})")
    
    height, width = annotated_frame.shape[:2]
    text_scale = max(1.0, width/1280)
    text_thickness = max(2, int(width/640))
    
    text_size = cv2.getTextSize(f"Command: {command_name}", cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
    cv2.rectangle(annotated_frame, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), (0, 0, 0), -1)
    
    cv2.putText(annotated_frame, f"Command: {command_name}", (15, 15 + text_size[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, text_thickness - 1)[0]
    cv2.rectangle(annotated_frame, (width - time_text_size[0] - 20, 10), 
                 (width - 10, 10 + time_text_size[1] + 10), (0, 0, 0), -1)
    cv2.putText(annotated_frame, timestamp, (width - time_text_size[0] - 15, 15 + time_text_size[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (0, 255, 255), text_thickness - 1)
    
    file_name = f"{save_dir}/frame_{frame_number:04d}_{command_name}.jpg"
    
    cv2.imwrite(file_name, annotated_frame)
    print(f"Saved frame with command '{command_name}' to {file_name}")
    
    return annotated_frame

def find_nearest_path_point(current_position, path):
    if not path:
        return None
        
    nearest_point = None
    min_distance = float('inf')
    nearest_index = 0
    
    for i, point in enumerate(path):
        distance = np.sqrt((point[0] - current_position[0])**2 + (point[1] - current_position[1])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
            nearest_index = i
            
    return nearest_point, nearest_index, min_distance        

def determine_command_with_directions(current_position, path, current_direction, last_command=None):
    if not path or len(path) < 2:
        print(len(path))
        return 's', False  # Return a tuple with default values
    
    nearest_point, nearest_idx, min_dist = find_nearest_path_point(current_position, path)
    
    if min_dist > 5:
        print(f"Car is off path (distance: {min_dist:.2f}). Attempting to rejoin path.")
        # Calculate the target direction
        dy = nearest_point[0] - current_position[0]
        dx = nearest_point[1] - current_position[1]
        target_direction = get_target_direction(dy, dx)
        
        # Return the turn command with a tuple
        return get_turn_command(current_direction, target_direction), False
    
    if nearest_idx >= len(path) - 30:
        print("Close to goal, stopping.")
        return 's', True  
    
    lookahead_idx = min(nearest_idx + 5, len(path) - 1)
    target_point = path[lookahead_idx]
    
    dy = target_point[0] - current_position[0]
    dx = target_point[1] - current_position[1]
    target_direction = get_target_direction(dy, dx)
    turn_command = get_turn_command(current_direction, target_direction)
    
    if last_command == 'l' and turn_command == 'r':
        print("Skipping right turn after left. Moving forward instead.")
        return 'f', False
    elif last_command == 'r' and turn_command == 'l':
        print("Skipping left turn after right. Moving forward instead.")
        return 'f', False
    
    return turn_command, False

# Add these helper functions at an appropriate location in your code (e.g., after DIR_TO_VEC)

def is_line_free(occupancy_grid, start_pos, direction, distance=1):
    """
    Check if all cells along a line from the starting position in a given direction 
    up to a specified distance are free.
    
    Parameters:
    - occupancy_grid: 2D numpy array where 0 represents a free cell.
    - start_pos: tuple (y, x) representing the starting position in the grid.
    - direction: tuple (dy, dx) indicating the direction to check.
      For instance, if you want to check forward relative to the car, and your carï¿½s 
      current direction is encoded by DIR_TO_VEC, use that (e.g., (-1, 0) for NORTH).
    - distance: integer specifying how many cells ahead to check.
    
    Returns:
    - True if all cells along the path are free; otherwise, False.
    """
    y, x = start_pos
    dy, dx = direction
    for d in range(1, distance + 1):
        new_y = y + d * dy
        new_x = x + d * dx
        # Check grid boundaries
        if new_y < 0 or new_y >= occupancy_grid.shape[0] or new_x < 0 or new_x >= occupancy_grid.shape[1]:
            return False
        if occupancy_grid[new_y, new_x] != 0:
            return False
    return True


def check_front_back(occupancy_grid, car_position, current_direction):
    """
    Given the occupancy grid, the current position and direction of the car,
    returns two booleans indicating whether the cell in front and the cell behind are free.
    """

    forward_direction = DIR_TO_VEC[current_direction]
    backward_direction = (-forward_direction[0], -forward_direction[1])
    front_free = is_line_free(occupancy_grid, car_position, forward_direction, distance=5)
    back_free = is_line_free(occupancy_grid, car_position, backward_direction, distance=5)
    return front_free, back_free


def get_target_direction(dy, dx):
    """
    Get the target direction based on the movement vector.
    
    Parameters:
    dy (float): Change in y-coordinate
    dx (float): Change in x-coordinate
    
    Returns:
    str: Target direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
    """
    # Find the most significant component (greatest magnitude)
    if abs(dy) > abs(dx):
        # Vertical movement is more significant
        if dy < 0:
            return 'NORTH'  # Moving up
        else:
            return 'SOUTH'  # Moving down
    else:
        # Horizontal movement is more significant
        if dx > 0:
            return 'EAST'   # Moving right
        else:
            return 'WEST'   # Moving left

def get_turn_command(current_direction, target_direction):
    """
    Determine the command to turn from current direction to target direction.
    
    Parameters:
    current_direction (str): Current direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
    target_direction (str): Target direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
    
    Returns:
    str: Command to execute ('f', 'l', 'r')
    """
    if current_direction == target_direction:
        return 'f'  # Already facing the right direction
    
    current_idx = DIRECTIONS.index(current_direction)
    target_idx = DIRECTIONS.index(target_direction)
    
    # Calculate the shortest turn
    diff = (target_idx - current_idx) % 4
    
    if diff == 1:
        return 'r'  # Turn right
    elif diff == 3:
        return 'l'  # Turn left
    else:  # diff == 2
        # 180 degree turn, let's choose right arbitrarily
        return 'r'

# Update the main function
def main():
    ser = init_serial()
    if ser is None:
        print("Failed to initialize serial connection. Exiting.")
        return
    
    last_command = 's'
    current_direction = 'NORTH'  # Default starting direction
    frame_number = 0
    is_first_command = True
    previous_positions = []
    current_path = None
    fire_goal_position = None
    initial_car_position = None

    while True:
        frame = get_frame()

        detected, position, _ = detect_fire(frame, FIRE_WEIGHTS_PATH, confidence_threshold=temp_threshold)
        if detected and position is not None:
            grid_height, grid_width = 50, 50
            frame_height, frame_width = frame.shape[:2]
            # Assuming position[0] is y and position[1] is x (see next section for coordinate checks)
            grid_y = int(position[0] * grid_height / frame_height)
            grid_x = int(position[1] * grid_width / frame_width)
            fire_goal_position = (grid_y, grid_x)
            print(f"Fire detected at frame position {position}, mapped to grid: {fire_goal_position}")
        else:
            # Reset the goal if no fire is detected
            fire_goal_position = None
            print("No fire detected this frame.")


        goal_position = fire_goal_position if fire_goal_position else None

        last_command, segmented_frame, occupancy_grid, current_path, current_direction, is_first_command, previous_positions, goal_position, initial_car_position = process_frame(
            frame, last_command, ser, current_direction, is_first_command, frame_number, 
            previous_positions, current_path, goal_position, initial_car_position
        )
        
        frame_number += 1
        
        api.latest_results = {
            'segmented_frame': segmented_frame,
            'occupancy_grid': occupancy_grid,
            'path': current_path,
            'last_command': last_command,
            'car_position': previous_positions[-1] if previous_positions else None,
            'direction': current_direction
        }
        
        cv2.imshow('Car Navigation', segmented_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_command(ser, 's')
            save_frame_with_command(segmented_frame, 's', frame_number)
            break
    
    cv2.destroyAllWindows()
    ser.close()

def run_flask_app():
    api.app.run(debug=True, use_reloader=False, port=5001)

if __name__ == "__main__":
    api.latest_results = {
        'segmented_frame': None,
        'occupancy_grid': None,
        'path': None,
        'last_command': 's',
        'car_position': None,
        'heading': 0
    }
    
    print(f"Frames will be saved to directory: {os.path.abspath(save_dir)}")
    
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    main()
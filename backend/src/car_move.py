#car_move.py
import threading
import time
import cv2
import serial
import numpy as np
from camera import get_frame
from object_detection import segment_objects_and_create_grid
from path import generate_waypoints_with_velocity
from func import path_following_thread

FORWARD = "f"
BACKWARD = "b"
LEFT = "r"
RIGHT = "l"
STOP = "s"
AUTO = "a"

DISTANCE_THRESHOLD = 3
TURN_THRESHOLD = 0.7
POSITION_UPDATE_INTERVAL = 0.5

class CarController:
    def __init__(self, serial_port='/dev/serial0', baud_rate=9600):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.is_connected = False
        self.current_command = STOP
        self.auto_mode = False
        self.connect()
        
    def connect(self):
        try:
            if self.ser is not None and self.ser.isOpen():
                self.ser.close()
                time.sleep(0.5)
            self.ser = serial.Serial(self.serial_port, self.baud_rate)
            time.sleep(2)
            self.is_connected = True
            print(f"Connected to car at {self.serial_port}")
        except Exception as e:
            print(f"Failed to connect to car: {e}")
            self.is_connected = False
            
    def send_command(self, command):
        if not self.is_connected:
            print("Not connected to car. Attempting to reconnect...")
            self.connect()
            if not self.is_connected:
                return False
        try:
            self.ser.write(command.encode())
            self.ser.flush()
            time.sleep(0.1)
            self.current_command = command
            print(f"Command sent: {command}")
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            self.is_connected = False
            return False
            
    def stop(self):
        return self.send_command(STOP)
        
    def forward(self):
        return self.send_command(FORWARD)
        
    def backward(self):
        return self.send_command(BACKWARD)
        
    def left(self):
        return self.send_command(LEFT)
        
    def right(self):
        return self.send_command(RIGHT)
        
    def set_auto_mode(self, enabled=True):
        if enabled:
            result = self.send_command(AUTO)
        else:
            result = self.send_command(STOP)
        self.auto_mode = enabled and result
        return result
        
    def close(self):
        if self.is_connected and self.ser:
            self.stop()
            self.ser.close()
            self.is_connected = False
            print("Connection to car closed")

class PathFollower:
    def __init__(self, car_controller, weights_path=None, labels_path=None):
        self.car = car_controller
        self.current_path = None
        self.current_waypoints = None
        self.current_waypoint_index = 0
        self.is_following_path = False
        self.actual_position = None
        self.estimated_position = (0, 0)
        self.orientation = 0
        self.stop_before_goal = True
        self.distance_to_stop = 2
        self.last_position_update = 0
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.confidence_threshold = 0.75
        self.position_lock = threading.Lock()
        self.last_command_time = time.time()
        self.command_cooldown = 0.5
        self.path_deviation_threshold = 5
        self.consecutive_position_failures = 0
        self.max_position_failures = 5
        self.grid_rows = 20
        self.grid_cols = 20
        
    def set_path(self, path):
        if path and len(path) > 1:
            self.current_path = path
            self.current_waypoints = generate_waypoints_with_velocity(path)
            self.current_waypoint_index = 0
            print(f"New path set with {len(path)} points")
            return True
        else:
            print("Invalid path - not setting")
            return False
    
    def set_detection_params(self, weights_path, labels_path, confidence_threshold=0.75, grid_rows=20, grid_cols=20):
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
            
    def start_following(self):
        if not self.current_path:
            print("No path set to follow")
            return False
        self.update_position_from_camera()
        if self.actual_position:
            self.estimated_position = self.actual_position
        else:
            self.estimated_position = self.current_path[0]
        self.is_following_path = True
        self.car.forward()
        print("Started path following - initial forward movement")
        return True
        
    def stop_following(self):
        self.is_following_path = False
        self.car.stop()
        print("Stopped path following")
    
    def update_position_from_camera(self):
        try:
            with self.position_lock:
                frame = get_frame()
                if frame is None or frame.size == 0:
                    print("Failed to get frame from camera")
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
            
    def get_current_position(self):
        if self.actual_position and time.time() - self.last_position_update < 5:
            return self.actual_position
        return self.estimated_position
            
    def update_position(self):
        if time.time() - self.last_position_update > POSITION_UPDATE_INTERVAL:
            camera_update_success = self.update_position_from_camera()
            if camera_update_success:
                self.estimated_position = self.actual_position
            elif self.consecutive_position_failures > self.max_position_failures:
                print(f"WARNING: Failed to detect car position {self.consecutive_position_failures} times in a row")
                if self.consecutive_position_failures > self.max_position_failures * 2:
                    print("EMERGENCY: Car position lost for too long. Stopping car.")
                    self.stop_following()
                    return
        if self.is_following_path and self.current_path and not (self.actual_position and time.time() - self.last_position_update < 0.2):
            curr_pos = self.get_current_position()
            if self.car.current_command == FORWARD:
                orientations = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                dy, dx = orientations[self.orientation]
                new_y = curr_pos[0] + dy * 0.5
                new_x = curr_pos[1] + dx * 0.5
                self.estimated_position = (new_y, new_x)
    
    def calculate_required_turn(self, current_angle, target_angle):
        current_angle = current_angle % 360
        target_angle = target_angle % 360
        clockwise = (target_angle - current_angle) % 360
        counter_clockwise = (current_angle - target_angle) % 360
        if clockwise <= counter_clockwise:
            return RIGHT
        else:
            return LEFT
    
    def get_path_deviation(self):
        if not self.current_path or self.current_waypoint_index >= len(self.current_waypoints):
            return float('inf')
        current_pos = self.get_current_position()
        target_y, target_x, _ = self.current_waypoints[self.current_waypoint_index]
        if self.current_waypoint_index > 0:
            prev_y, prev_x, _ = self.current_waypoints[self.current_waypoint_index - 1]
            line_length = np.sqrt((target_y - prev_y)**2 + (target_x - prev_x)**2)
            if line_length < 0.1:
                return np.sqrt((current_pos[0] - target_y)**2 + (current_pos[1] - target_x)**2)
            cross_product = abs((target_x - prev_x)*(prev_y - current_pos[0]) - (prev_x - current_pos[1])*(target_y - prev_y))
            return cross_product / line_length
        else:
            return np.sqrt((current_pos[0] - target_y)**2 + (current_pos[1] - target_x)**2)
    
    def should_recalculate_path(self):
        deviation = self.get_path_deviation()
        return deviation > self.path_deviation_threshold
    
    def get_next_command(self):
        if not self.is_following_path or not self.current_waypoints:
            return STOP
        if self.current_waypoint_index >= len(self.current_waypoints):
            print("Reached end of waypoints")
            return STOP
        if time.time() - self.last_command_time < self.command_cooldown:
            return self.car.current_command
        current_pos = self.get_current_position()
        next_y, next_x, velocity = self.current_waypoints[self.current_waypoint_index]
        dy = next_y - current_pos[0]
        dx = next_x - current_pos[1]
        distance = np.sqrt(dy**2 + dx**2)
        print(f"Current position: {current_pos}, Next waypoint: ({next_y}, {next_x}), Distance: {distance:.2f}")
        if self.should_recalculate_path():
            print(f"WARNING: Car has deviated from path by {self.get_path_deviation():.2f} units. Consider recalculating.")
        if self.stop_before_goal and self.current_waypoint_index == len(self.current_waypoints) - 1:
            if distance <= self.distance_to_stop:
                print(f"Stopping {self.distance_to_stop} cells before reaching the fire")
                self.current_waypoint_index += 1
                self.last_command_time = time.time()
                return STOP
        if distance < DISTANCE_THRESHOLD:
            print(f"Reached waypoint {self.current_waypoint_index}")
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.current_waypoints):
                print("Reached end of path")
                self.last_command_time = time.time()
                return STOP
            next_y, next_x, velocity = self.current_waypoints[self.current_waypoint_index]
            dy = next_y - current_pos[0]
            dx = next_x - current_pos[1]
        angle_to_target = np.degrees(np.arctan2(dy, dx)) % 360
        orientation_angles = [0, 90, 180, 270]
        current_orientation_angle = orientation_angles[self.orientation]
        angle_diff = (angle_to_target - current_orientation_angle) % 360
        command = None
        if angle_diff < 45 or angle_diff > 315:
            command = FORWARD
        elif 45 <= angle_diff < 135:
            self.orientation = (self.orientation + 1) % 4
            command = RIGHT
        elif 225 <= angle_diff < 315:
            self.orientation = (self.orientation - 1) % 4
            command = LEFT
        else:
            self.orientation = (self.orientation + 2) % 4
            command = RIGHT
        self.last_command_time = time.time()
        return command
            
    def follow_path_step(self):
        if not self.is_following_path:
            return
        self.update_position()
        command = self.get_next_command()
        if command != self.car.current_command:
            self.car.send_command(command)
        if self.current_waypoint_index >= len(self.current_waypoints):
            print("Path completed")
            self.stop_following()
car_controller = CarController()
path_follower = PathFollower(car_controller)
path_following_thread = threading.Thread(target=path_following_thread, daemon=True)
path_following_thread.start()
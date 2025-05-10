#path.py
import numpy as np
import heapq
import cv2
import random
from collections import defaultdict

CAR_WIDTH_INCHES = 8
GRID_CELL_SIZE_INCHES = 1
CAR_WIDTH_CELLS = int(np.ceil(CAR_WIDTH_INCHES / GRID_CELL_SIZE_INCHES))
SAFETY_MARGIN = 3

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def diagonal_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def preprocess_grid_for_car(occupancy_grid):
    expanded_grid = occupancy_grid.copy()
    height, width = occupancy_grid.shape
    expansion_radius = CAR_WIDTH_CELLS // 2 + SAFETY_MARGIN
    obstacle_coords = np.where(occupancy_grid == 1)
    obstacle_points = list(zip(obstacle_coords[0], obstacle_coords[1]))
    for y, x in obstacle_points:
        for dy in range(-expansion_radius, expansion_radius + 1):
            for dx in range(-expansion_radius, expansion_radius + 1):
                if np.sqrt(dy**2 + dx**2) <= expansion_radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        expanded_grid[ny, nx] = 1
    return expanded_grid

def smooth_path(path, occupancy_grid, smoothing_factor=0.5):
    if len(path) <= 2:
        return path
    smoothed_path = [path[0]]
    for i in range(1, len(path) - 1):
        prev_point = np.array(smoothed_path[-1])
        current_point = np.array(path[i])
        next_point = np.array(path[i + 1])
        vector_to_next = next_point - current_point
        vector_from_prev = current_point - prev_point
        smoothed_point = current_point + smoothing_factor * (vector_to_next + vector_from_prev) / 2
        y, x = int(smoothed_point[0]), int(smoothed_point[1])
        if 0 <= y < occupancy_grid.shape[0] and 0 <= x < occupancy_grid.shape[1] and occupancy_grid[y, x] == 0:
            prev_y, prev_x = smoothed_path[-1]
            if abs(y - prev_y) > 1 or abs(x - prev_x) > 1:
                points_between = get_line_points(prev_y, prev_x, y, x)
                for py, px in points_between[1:-1]:
                    if 0 <= py < occupancy_grid.shape[0] and 0 <= px < occupancy_grid.shape[1] and occupancy_grid[py, px] == 0:
                        smoothed_path.append((py, px))
            smoothed_path.append((y, x))
        else:
            y, x = path[i]
            prev_y, prev_x = smoothed_path[-1]
            if abs(y - prev_y) > 1 or abs(x - prev_x) > 1:
                points_between = get_line_points(prev_y, prev_x, y, x)
                for py, px in points_between[1:-1]:
                    if 0 <= py < occupancy_grid.shape[0] and 0 <= px < occupancy_grid.shape[1] and occupancy_grid[py, px] == 0:
                        smoothed_path.append((py, px))
            smoothed_path.append((y, x))
    final_y, final_x = path[-1]
    prev_y, prev_x = smoothed_path[-1]
    if abs(final_y - prev_y) > 1 or abs(final_x - prev_x) > 1:
        points_between = get_line_points(prev_y, prev_x, final_y, final_x)
        for py, px in points_between[1:-1]:
            if 0 <= py < occupancy_grid.shape[0] and 0 <= px < occupancy_grid.shape[1] and occupancy_grid[py, px] == 0:
                smoothed_path.append((py, px))
    smoothed_path.append(path[-1])
    return smoothed_path

def get_line_points(y0, x0, y1, x1):
    points = []
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    return points

def calculate_path_curvature(path):
    if len(path) < 3:
        return [0] * len(path)
    curvatures = [0]
    for i in range(1, len(path) - 1):
        prev = np.array(path[i - 1])
        current = np.array(path[i])
        next_point = np.array(path[i + 1])
        v1 = current - prev
        v2 = next_point - current
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product < 1e-10:
            curvatures.append(0)
        else:
            angle = np.arccos(min(1, max(-1, dot_product / norm_product)))
            curvatures.append(angle / np.pi)
    curvatures.append(0)
    return curvatures

def generate_waypoints_with_velocity(path, max_velocity=2.0, deceleration_factor=0.5):
    if not path:
        return []
    curvatures = calculate_path_curvature(path)
    waypoints = []
    for i, (y, x) in enumerate(path):
        velocity = max_velocity * (1 - curvatures[i] * deceleration_factor)
        waypoints.append((y, x, velocity))
    return waypoints

def a_star_car_pathfinding(occupancy_grid, start, goal, heuristic_type='euclidean', allow_diagonal=False):
    if not isinstance(occupancy_grid, np.ndarray):
        occupancy_grid = np.array(occupancy_grid)
    car_grid = preprocess_grid_for_car(occupancy_grid)
    height, width = car_grid.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width and 
            0 <= goal[0] < height and 0 <= goal[1] < width):
        return []
    car_grid[start] = 0
    if car_grid[goal] == 1:
        goal_free = find_nearest_free_cell(car_grid, goal)
        if goal_free:
            goal = goal_free
        else:
            return []
    if heuristic_type == 'manhattan':
        heuristic_func = manhattan_distance
    elif heuristic_type == 'euclidean':
        heuristic_func = euclidean_distance
    elif heuristic_type == 'diagonal':
        heuristic_func = diagonal_distance
    else:
        heuristic_func = manhattan_distance
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move_costs = [1, 1, 1, 1]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_func(start, goal)}
    closed_set = set()
    iterations = 0
    max_iterations = height * width * 2
    while open_set and iterations < max_iterations:
        iterations += 1
        current_f, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            reversed_path = path[::-1]
            return reversed_path
        for i, direction in enumerate(directions):
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                if neighbor in closed_set:
                    continue
                if car_grid[neighbor] == 0:
                    tentative_g_score = g_score[current] + move_costs[i]
                    if current in came_from:
                        prev = came_from[current]
                        prev_direction = (current[0] - prev[0], current[1] - prev[1])
                        if prev_direction != direction:
                            tentative_g_score += 0.5
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def find_optimal_car_path(occupancy_grid, start, goal, method='a_star', allow_diagonal=False):
    if method == 'a_star':
        path = a_star_car_pathfinding(occupancy_grid, start, goal, 'manhattan', False)
        if path:
            waypoints = generate_waypoints_with_velocity(path)
            return waypoints
        return None
    elif method == 'q_learning':
        return None
    else:
        return None

def find_nearest_free_cell(grid, position, max_search_radius=10):
    y, x = position
    height, width = grid.shape
    for radius in range(1, max_search_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) == radius or abs(dx) == radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == 0:
                        return (ny, nx)
    return None
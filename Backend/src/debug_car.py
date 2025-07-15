import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

# Import functions from your path.py file
# Uncomment this line and update the path to import your actual module
# from path import preprocess_grid_for_car, a_star_car_pathfinding, generate_waypoints_with_velocity, calculate_path_curvature

# For demonstration, I'll redefine the key functions needed
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

def visualize_velocity_path(grid, path, title="Car Path with Velocity", show_curvature=False, save_path=None):
    """
    Create a visualization of the path with color-coded velocities
    
    Args:
        grid: Occupancy grid where 1 represents obstacles
        path: List of (y, x) tuples representing the path
        title: Title for the plot
        show_curvature: Whether to show the curvature plot
        save_path: Path to save the image (if None, image is displayed)
    """
    if not path:
        print("No path to visualize")
        return
    
    # Convert to waypoints with velocity
    waypoints = generate_waypoints_with_velocity(path, max_velocity=2.0, deceleration_factor=0.5)
    
    # Extract data from waypoints
    y_coords = [wp[0] for wp in waypoints]
    x_coords = [wp[1] for wp in waypoints]
    velocities = [wp[2] for wp in waypoints]
    
    # Calculate curvatures for comparison/validation
    curvatures = calculate_path_curvature(path)
    
    # Create visualization
    if show_curvature:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Grid and path
    ax1.imshow(grid, cmap='binary', interpolation='none', origin='upper')
    ax1.set_title('Occupancy Grid with Path')
    
    # Create a colormap from red (slow) to green (fast)
    cmap = LinearSegmentedColormap.from_list('velocity_cmap', ['red', 'yellow', 'green'])
    
    # Normalize velocities to [0, 1] for coloring
    min_vel = min(velocities)
    max_vel = max(velocities)
    norm_velocities = [(v - min_vel) / (max_vel - min_vel) if max_vel > min_vel else 0.5 for v in velocities]
    
    # Plot the path points
    for i in range(len(path) - 1):
        ax1.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                 color=cmap(norm_velocities[i]), linewidth=3)
    
    # Mark start and end points
    ax1.plot(x_coords[0], y_coords[0], 'bo', markersize=10, label='Start')
    ax1.plot(x_coords[-1], y_coords[-1], 'mo', markersize=10, label='Goal')
    ax1.legend()
    
    # Plot 2: Velocity profile along the path
    path_indices = list(range(len(path)))
    ax2.plot(path_indices, velocities, 'b-', linewidth=2)
    ax2.set_title('Velocity Profile Along Path')
    ax2.set_xlabel('Path Point Index')
    ax2.set_ylabel('Velocity')
    ax2.grid(True)
    
    # Add velocity points with colors
    scatter = ax2.scatter(path_indices, velocities, c=velocities, cmap=cmap, 
                          s=50, zorder=3, norm=plt.Normalize(min_vel, max_vel))
    plt.colorbar(scatter, ax=ax2, label='Velocity')
    
    # Plot 3: Curvature (optional)
    if show_curvature:
        ax3.plot(path_indices, curvatures, 'r-', linewidth=2)
        ax3.set_title('Path Curvature')
        ax3.set_xlabel('Path Point Index')
        ax3.set_ylabel('Curvature')
        ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

def create_example_grid_and_path(grid_size=20, obstacle_density=0.1, path_length=15):
    """Create an example grid and path for demonstration"""
    # Create grid with random obstacles
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    np.random.seed(42)  # For reproducibility
    
    # Add some random obstacles
    for _ in range(int(grid_size * grid_size * obstacle_density)):
        y, x = np.random.randint(0, grid_size, 2)
        grid[y, x] = 1
    
    # Add some structure to obstacles
    grid[5:8, 5:10] = 1  # Rectangular obstacle
    grid[12:15, 15:18] = 1  # Another obstacle
    
    # Create a hand-crafted path that includes straight segments and curves
    # Start with a straight path
    path = [(grid_size-2, 2)]
    for i in range(1, 6):
        path.append((grid_size-2-i, 2))
    
    # Add a curve
    path.append((grid_size-8, 3))
    path.append((grid_size-9, 4))
    path.append((grid_size-10, 5))
    
    # Another straight segment
    for i in range(6, 10):
        path.append((grid_size-10, i))
    
    # Another curve
    path.append((grid_size-11, 11))
    path.append((grid_size-12, 12))
    
    # Final straight segment
    path.append((grid_size-13, 12))
    path.append((grid_size-14, 12))
    
    return grid, path

def visualize_a_star_path(grid, start, goal, save_path=None):
    """Visualize A* path finding with velocity color-coding"""
    # This would normally import from path.py
    # For demo, we'll just show a placeholder function
    def a_star_pathfinding(grid, start, goal):
        """Placeholder for the actual A* algorithm from your path.py"""
        # This is just a simplified example - normally you'd call your actual function
        path = []
        current = start
        path.append(current)
        
        # Generate a simple path between start and goal
        while current != goal:
            y, x = current
            if y < goal[0]:
                y += 1
            elif y > goal[0]:
                y -= 1
            elif x < goal[1]:
                x += 1
            elif x > goal[1]:
                x -= 1
            
            # Add some randomness for curves
            if np.random.random() > 0.7:
                if y != goal[0]:
                    if np.random.random() > 0.5:
                        x += np.random.choice([-1, 0, 1])
                if x != goal[1]:
                    if np.random.random() > 0.5:
                        y += np.random.choice([-1, 0, 1])
            
            # Keep in bounds
            y = max(0, min(y, grid.shape[0]-1))
            x = max(0, min(x, grid.shape[1]-1))
            
            # Skip if obstacle
            if grid[y, x] == 1:
                continue
                
            current = (y, x)
            path.append(current)
        
        return path
    
    # Get path using A*
    path = a_star_pathfinding(grid, start, goal)
    
    # Visualize the path with velocity information
    visualize_velocity_path(grid, path, title="A* Path with Velocity", show_curvature=True, save_path=save_path)
    
    return path

def main():
    # Example 1: Visualize a simple pre-defined path
    print("Example 1: Predefined path with velocity visualization")
    grid, path = create_example_grid_and_path(grid_size=20)
    visualize_velocity_path(grid, path, show_curvature=True)
    
    # Example 2: Visualize an A* path
    print("\nExample 2: A* pathfinding with velocity visualization")
    grid = np.zeros((30, 30), dtype=np.uint8)
    
    # Add obstacles
    grid[5:15, 10:12] = 1  # Vertical wall
    grid[20:25, 15:25] = 1  # Large obstacle
    grid[7:12, 20:22] = 1  # Another obstacle
    
    start = (5, 5)
    goal = (25, 25)
    
    visualize_a_star_path(grid, start, goal)
    
    print("\nDone! Close the visualization windows to exit.")

if __name__ == "__main__":
    main()
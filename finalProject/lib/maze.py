import pygame
import random
# TODO: 3 obstacle types: user-placed, stationary and in-motion

def randomize_obstacles(x: int, y: int, num_obstacles):
    """
    Generate random obstacle coordinates
    
    Args:
    x (int): Screen width
    y (int): Screen height
    num_obstacles (int, optional): Number of obstacles to generate. 
                                   Defaults to a proportion of screen size if not specified.
    
    Returns:
    list: List of (x, y) coordinates for obstacles
    """
    if num_obstacles is None:
        num_obstacles = (x * y) // (50 * 50)
    
    obstacles = []
    grid_size = 10
    
    for _ in range(num_obstacles):
        # Generate coordinates aligned to a grid
        obstacle_x = random.randint(0, (x - grid_size) // grid_size) * grid_size
        obstacle_y = random.randint(0, (y - grid_size) // grid_size) * grid_size
        obstacles.append((obstacle_x, obstacle_y))

    return obstacles

# INFO: create a maze with 5px size solid obstacles
def draw(screen: pygame.SurfaceType, obstacles, color: str="white"):
    rects = list((pygame.Rect(x, y, 10, 10) for x, y in obstacles))
    for rect in rects:
        pygame.draw.rect(screen, color, rect)
    return rects

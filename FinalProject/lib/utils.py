from typing import Tuple
import math

def getNeighbors(w: int, h: int):
    """
    Returns a function to compute the valid neighbors of a given cell in a 2D grid.

    Parameters:
        w (int): The width of the grid.
        h (int): The height of the grid.

    Returns:
        function: A function that takes x (column index) and y (row index) as inputs 
                  and returns a list of valid neighboring coordinates within the grid 
                  boundaries.
    """
    def __inner__(x: int, y: int):
        nonlocal w, h
        neighbors = [
            (x - 1,     y),
            (x - 1, y + 1),
            (x    , y + 1),
            (x + 1, y + 1),
            (x + 1,     y),
            (x + 1, y - 1),
            (x    , y - 1),
            (x - 1, y - 1),
        ]
        return [ (a, b) for (a, b) in neighbors if 0 <= a < w and 0 <= b < h ]
    return __inner__


def manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]):
    """
    Returns the manhattan distance between two coordinates

    Parameters:
        p1 (Tuple[int, int]): The (x, y) coordinates of the first point.
        p2 (Tuple[int, int]): The (x, y) coordinates of the second point.

    Returns:
        int: The Manhattan distance between the two points, defined as the 
             sum of the absolute differences of their x and y coordinates.
    """
    x, y = p1
    a, b = p2
    return abs(x - a) + abs(y - b)


def euclid_dist(p1: Tuple[int, int], p2: Tuple[int, int]):
    """
    Computes the Euclidean distance between two points in a 2D plane.

    Parameters:
        p1 (Tuple[int, int]): The (x, y) coordinates of the first point.
        p2 (Tuple[int, int]): The (x, y) coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points, calculated as the
               square root of the sum of squared differences of their coordinates.
    """
    x, y = p1
    a, b = p2
    return math.sqrt((x - a) ** 2 + (y - b) ** 2)

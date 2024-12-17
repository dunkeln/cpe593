from typing import List, Tuple, Callable
from .utils import *
import heapq

# INFO: utility functions

manhattan_dist((10, 20), (10, 20))

def fibonacci_heap():
    # TODO: implement the damn heap!!!
    pass

# INFO: A* fibonacci heap
def astar(
        grid: Tuple[int, int],
        start: Tuple[int, int],
        end: Tuple[int, int],
        heap=heapq,
        distance_metric=euclid_dist
):
    frontier = []
    heap.heappush(start, 0) # INFO: initializing with start pos and f_score = 0
    closed = {}
    costs = {}

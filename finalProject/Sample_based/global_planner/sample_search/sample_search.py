import numpy as np
from itertools import combinations
import math
import sys, os
import torch
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils import Env, Node, Plot, Planner

class SampleSearcher(Planner):
    '''
    Base class for planner based on sample searching.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, delta: float=0.5) -> None:
        super().__init__(start, goal, env)
        # inflation bias
        self.delta = delta

    def isCollision(self, node1: Node, node2: Node) -> bool:
        '''
        Judge collision when moving from node1 to node2.

        Parameters
        ----------
        node1, node2: Node

        Return
        ----------
        collision: bool
            True if collision exists else False
        '''
        if self.isInsideObs(node1) or self.isInsideObs(node2):
            return True

        if self.isIntersectObs(node1, node2):
            return True

        return False

    def isInsideObs(self, node: Node) -> bool:
        ''' Determine whether the node is inside obstacles'''
        return self.env.occupancy_grid_map[round(node.x), round(node.y)]

    def isIntersectObs(self, node1: Node, node2: Node) -> bool:
        '''Determine whether the segment (node1---node2) intersects with the obstacles'''
        Seg_AB = torch.tensor([node1.current, node2.current])
        return self._Is_Seg_Ingersection_1toM(Seg_AB, self.env.Obstacle_Segments)

    def _cross_product_N2_1(self, Vss, U):
        '''计算向量Vss和U的交叉积 (x0*y1-x1*y0)
            Vss = torch.tensor((N,2,2))
            U = torch.tensor([x0, y0]), shape=(2,)
            Output = torch.tensor((N,2))'''
        return Vss[:, :, 0] * U[1] - Vss[:, :, 1] * U[0]

    def _cross_product_N2_N1(self, Vss, Us):
        '''计算向量Vss和Us的交叉积 (x0*y1-x1*y0)
            Vss = torch.tensor((N,2,2))
            Us = torch.tensor((N,2))
            Output = torch.tensor((N,2)) '''
        return Vss[:, :, 0] * Us[:, 1, None] - Vss[:, :, 1] * Us[:, 0, None]

    def _Is_Seg_Ingersection_1toM(self, Seg_AB, Segs) -> bool:
        '''利用[交叉积-跨立实验]判断线段AB与线段集Segs是否相交
            Seg_AB = torch.tensor([[Xa,Ya], [Xb,Yb]]), (2,2)
            Segs = torch.tensor( [ [[X00,Y00], [X01,Y01]], [[X10,Y10], [X11,Y11]], ...] ), (N,2,2)'''

        V_AB = Seg_AB[1] - Seg_AB[0]  # (2,)
        V_A_Segs = Segs - Seg_AB[0]  # (N,2,2)
        Flag1 = self._cross_product_N2_1(V_A_Segs, V_AB).prod(dim=-1) < 0  # (N,), Segs中各线段的两端点 是否跨立 线段AB 两侧

        V_SA_SB = Seg_AB - Segs  # (N,2,2)
        V_Seg = Segs[:, 1] - Segs[:, 0]  # (N,2)
        Flag2 = self._cross_product_N2_N1(V_SA_SB, V_Seg).prod(dim=-1) < 0  # (N,), AB两点 是否跨立 Segs中各线段 两侧

        Intersection = Flag1 * Flag2  # 两次跨立，则相交, shape=(N,)

        return Intersection.any() # bool


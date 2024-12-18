import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import torch
import time

from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement


from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.best_first import BestFirst
from pathfinding.finder.bi_a_star import BiAStarFinder
from pathfinding.finder.breadth_first import BreadthFirstFinder
from pathfinding.finder.msp import MinimumSpanningTree

class PathPlanningEnv():
    def __init__(self, PlannerIdx: int, Obs_Type: str):
        '''PlannerIdx: 0~5;  Obs_Type: CMO/RMO'''

        # Path Planners Init:
        if PlannerIdx == 0: self.planner = AStarFinder(diagonal_movement=DiagonalMovement.always)
        if PlannerIdx == 1: self.planner = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
        if PlannerIdx == 2: self.planner = BestFirst(diagonal_movement=DiagonalMovement.always)
        if PlannerIdx == 3: self.planner = BiAStarFinder(diagonal_movement=DiagonalMovement.always)
        if PlannerIdx == 4: self.planner = BreadthFirstFinder(diagonal_movement=DiagonalMovement.always)
        if PlannerIdx == 5: self.planner = MinimumSpanningTree(diagonal_movement=DiagonalMovement.always)
        name = ['A*', 'Dijkstra', 'BestFirst', 'Bi-directional A*', 'Breadth First Search', 'Minimum Spanning Tree']
        print(f'Planner Name:{name[PlannerIdx]}')

        # Map Init:
        self.Obs_Type = Obs_Type
        if Obs_Type == 'CMO': # Consistently Randomly Moving Obstacles
            self.Generated_Obstacle_Segments = torch.load('Generate_Obstacle_Segments/CMO_Obstacle_Segments.pt',map_location='cpu') #(M,2,2) or (4*O,2,2)
        elif Obs_Type == 'RMO': # Randomly Moving Obstacles
            self.Generated_Obstacle_Segments = torch.load('Generate_Obstacle_Segments/RMO_Obstacle_Segments.pt',map_location='cpu') #(M,2,2) or (4*O,2,2)
        else: print('Wrong Obstacle Type! Please choose "CMO" or "RMO".')

        self.window_size = 366
        self.grid_map = pygame.Surface((self.window_size, self.window_size)) #绘制障碍物（用0表示）,并转换为np.array, 用于路径规划
        self.O = self.Generated_Obstacle_Segments.shape[0] // 4  # 障碍物数量
        self.static_obs = 2
        self.dynamic_obs = self.O - self.static_obs
        self.Obs_X_limit = torch.tensor([46,300])
        self.Obs_Y_limit = torch.tensor([0, 366])
        self.Obstacle_V = 4 # 每个障碍物的最大速度
        self.Target_V = 3
        self.y_target_range = (5,360)
        self.Start_V = 6
        self.reach_trsd = 20 # 抵达终点的阈值
        self.travel_distance = 0
        self.render = False
        self.V_arrow = 15
        self.start_point = 20
        self.target_point = 330

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape) + low

    def Render_Init(self):
        self.render = True
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))

        # 绘制障碍物图层，用于Render
        self.map_pyg = pygame.Surface((self.window_size, self.window_size))

    def _Map_Init(self):
        self.collide = False
        # 起点/终点：
        self.x_start, self.y_start = self.start_point, self.start_point # 起点坐标
        self.x_target, self.y_target = self.target_point, self.target_point # 终点坐标
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # 障碍物坐标：
        self.Obs_Segments = self.Generated_Obstacle_Segments.clone()
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,4,2,2) #注意Grouped_Obs_Segments 和 Obs_Segments 是联动的

        # 障碍物速度:
        self.Obs_V = self._uniform_random(1, self.Obstacle_V, (self.O, 1, 1, 2))  # 每个障碍物的x方向速度和y方向速度
        self.Obs_V[-self.static_obs:]*=0 # 最后static_obs个障碍物不动
        self.Obs_V[0:int(self.dynamic_obs/2)] *= -1  # 前半障碍物速度反向

        if self.render:
            # 障碍物速度线
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.sum(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + self.V_arrow*self.Normed_Obs_V


    def _update_map(self):
        ''' 更新起点、终点、障碍物, 每次iterate后执行一次'''
        # 起点运动:
        next_x = self.waypoints[self.Start_V][0]
        next_y = self.waypoints[self.Start_V][1]
        self.travel_distance += ((self.x_start-next_x)**2 + (self.y_start-next_y)**2)**0.5
        self.x_start = next_x
        self.y_start = next_y
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # 终点运动:
        self.y_target += self.Target_V
        if self.y_target < self.y_target_range[0]:
            self.y_target = self.y_target_range[0]
            self.Target_V *= -1
        if self.y_target > self.y_target_range[1]:
            self.y_target = self.y_target_range[1]
            self.Target_V *= -1


        # 障碍物运动
        self.Grouped_Obs_Segments += self.Obs_V
        Flag_Vx = ((self.Grouped_Obs_Segments[:, :, :, 0] < self.Obs_X_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 0] > self.Obs_X_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vx, :, :, 0] *= -1
        Flag_Vy = ((self.Grouped_Obs_Segments[:, :, :, 1] < self.Obs_Y_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 1] > self.Obs_Y_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vy, :, :, 1] *= -1

        # 随机障碍物
        if self.Obs_Type == 'RMO':
            self.Obs_V += self._uniform_random(-1, 1, (self.O, 1, 1, 2))  # 每个障碍物的x方向速度和y方向速度
            self.Obs_V[-self.static_obs:]*=0 # 最后static_obs个障碍物不动
            self.Obs_V.clip_(-self.Obstacle_V, self.Obstacle_V) # 限速

        if self.render:
            # 生成障碍物速度箭头
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + self.V_arrow*self.Normed_Obs_V




    def _render_frame(self):
        self.map_pyg.fill((255, 255, 255))
        Grouped_Obs_Segments = self.Grouped_Obs_Segments.int().numpy()
        Grouped_Obs_center = self.Grouped_Obs_center.int().numpy()
        Grouped_Obs_Vend = self.Grouped_Obs_Vend.int().numpy()

        for _ in range(self.O):
            # 画障碍物
            obs_color = (50, 50, 50) if _ < self.dynamic_obs else (225, 100, 0)
            pygame.draw.polygon(self.map_pyg, obs_color, Grouped_Obs_Segments[_, :, 0, :])

            # 画障碍物速度
            if _ < self.dynamic_obs:
                pygame.draw.line(self.map_pyg, (255, 255, 0), Grouped_Obs_center[_], Grouped_Obs_Vend[_], width=2)
        self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

        # 画路径
        for point in self.waypoints:
            pygame.draw.circle(self.canvas, (0, 0, 255), point, 2)

        # 画起点、终点
        pygame.draw.circle(self.canvas, (255, 0, 0), (self.x_start, self.y_start), 5)  # 起点
        pygame.draw.circle(self.canvas, (0, 255, 0), (self.x_target, self.y_target), 5)  # 终点

        self.window.blit(self.canvas, self.map_pyg.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(0)


    def Dynamic_Plan(self):
        self._Map_Init()
        c, TPP = 0, 0
        while True:
            '''Step 1: Prepare the Grid Map with pygame'''
            self.grid_map.fill((1, 1, 1))
            Grouped_Obs_Segments = self.Grouped_Obs_Segments.int().numpy()
            for _ in range(self.O):
                pygame.draw.polygon(self.grid_map, (0,0,0), Grouped_Obs_Segments[_, :, 0, :]) # 将障碍物区域填充为0
            map_data_np = (pygame.surfarray.array3d(self.grid_map)[:, :, 0]).T
            if not map_data_np[self.y_start, self.x_start]: #Collide
                return 0, TPP
            grid_map = Grid(matrix=map_data_np)

            '''Step 2: Find path with Planner'''
            start = grid_map.node(self.x_start, self.y_start)
            target = grid_map.node(self.x_target, self.y_target)

            c += 1
            t0 = time.time()
            path, _ = self.planner.find_path(start, target, grid_map)
            TPP = TPP + ((time.time() - t0) - TPP) / c  # 增量法求平均 Time Per Planning

            self.waypoints = [[x,y] for x, y in path]

            '''Step 3: Update the map and Render'''
            if self.render: self._render_frame()
            self._update_map()


            '''Step 4: Stop Determination'''
            if self.d2target < self.reach_trsd:
                return self.travel_distance, TPP








import os,time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame,torch,scipy
import numpy as np
from utils import Map, SearchFactory, Node


class PathPlanningEnv():
    def __init__(self, PlannerIdx: int, Obs_Type: str):
        '''PlannerIdx: 0/1;  Obs_Type: CMO/RMO'''
        name = ['rrt', 'rrt_star']
        self.PlannerIdx = PlannerIdx
        self.planner_name = name[PlannerIdx]
        # print('Planner: '+self.planner_name)

        # Map Init:
        self.Obs_Type = Obs_Type
        self.Generated_Obstacle_Segments = torch.load(f'Generate_Obstacle_Segments/{Obs_Type}_Obstacle_Segments.pt',map_location='cpu') #(M,2,2) or (4*O,2,2)
        self.window_size = 366
        self.grid_map = pygame.Surface((self.window_size, self.window_size))
        self.O = self.Generated_Obstacle_Segments.shape[0] // 4  # Number of Obstacles
        self.static_obs = 2
        self.dynamic_obs = self.O - self.static_obs
        self.Obs_X_limit = torch.tensor([46,300])
        self.Obs_Y_limit = torch.tensor([0, 366])
        self.Obstacle_V = 4
        self.Target_V = 3
        self.y_target_range = (5,360)
        self.Start_V = 6
        self.reach_trsd = 20
        self.travel_distance = 0
        self.render = False
        self.V_arrow = 15
        self.x_start_init, self.y_start_init = 20,20
        self.x_target_init, self.y_target_init = 330,330

        self.search_factory = SearchFactory()
        self.kernel = np.array([[1., 1, 1],[1, 1, 1],[1, 1, 1]])

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape) + low

    def Render_Init(self, FPS):
        self.FPS = FPS
        self.render = True
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))

        # Render
        self.map_pyg = pygame.Surface((self.window_size, self.window_size))


    def _Map_Init(self):
        # starting point / end point:
        self.x_start, self.y_start = self.x_start_init, self.y_start_init
        self.x_target, self.y_target = self.x_target_init, self.y_target_init
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # obstacle coordinates:
        self.Obs_Segments = self.Generated_Obstacle_Segments.clone()
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,4,2,2) # Grouped_Obs_Segments, Obs_Segments

        # obstacle speed:
        self.Obs_V = self._uniform_random(1, self.Obstacle_V, (self.O, 1, 1, 2))  # The x-direction velocity and y-direction velocity of each obstacle
        self.Obs_V[-self.static_obs:]*=0 # static_obs
        self.Obs_V[0:int(self.dynamic_obs/2)] *= -1

        if self.render:
            # obstacle speed line
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.sum(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + self.V_arrow*self.Normed_Obs_V


    def _update_map(self):
        ''' Update the starting point, end point, and obstacles, and execute it once after each iterate '''
        if self.d2target<40: leadpoint = (self.x_target, self.y_target)
        else: leadpoint = self.waypoints[1]
        V_vector = torch.tensor(leadpoint) - torch.tensor([self.x_start, self.y_start])
        V_vector = V_vector / torch.sqrt(V_vector.pow(2).sum()+1e-6) # normalization
        self.x_start += V_vector[0].item() * self.Start_V
        self.y_start += V_vector[1].item() * self.Start_V
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target
        self.travel_distance += self.Start_V

        # finishing up
        self.y_target += self.Target_V
        if self.y_target < self.y_target_range[0]:
            self.y_target = self.y_target_range[0]
            self.Target_V *= -1
        if self.y_target > self.y_target_range[1]:
            self.y_target = self.y_target_range[1]
            self.Target_V *= -1

        # obstacle course
        self.Grouped_Obs_Segments += self.Obs_V
        Flag_Vx = ((self.Grouped_Obs_Segments[:, :, :, 0] < self.Obs_X_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 0] > self.Obs_X_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vx, :, :, 0] *= -1
        Flag_Vy = ((self.Grouped_Obs_Segments[:, :, :, 1] < self.Obs_Y_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 1] > self.Obs_Y_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vy, :, :, 1] *= -1

        # randomize obstacles
        if self.Obs_Type == 'RMO':
            self.Obs_V += self._uniform_random(-1, 1, (self.O, 1, 1, 2))  # the x and y velocities of each obstacle
            self.Obs_V[-self.static_obs:]*=0 # static_obs
            self.Obs_V.clip_(-self.Obstacle_V, self.Obstacle_V)

        if self.render:
            # speed limit
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + self.V_arrow*self.Normed_Obs_V


    def _render_frame(self):
        self.map_pyg.fill((255, 255, 255))
        Grouped_Obs_Segments = self.Grouped_Obs_Segments.int().numpy()
        Grouped_Obs_center = self.Grouped_Obs_center.int().numpy()
        Grouped_Obs_Vend = self.Grouped_Obs_Vend.int().numpy()

        for _ in range(self.O):
            # draw obstacles
            obs_color = (50, 50, 50) if _ < self.dynamic_obs else (225, 100, 0)
            pygame.draw.polygon(self.map_pyg, obs_color, Grouped_Obs_Segments[_, :, 0, :])

            # draw obstacle speed
            if _ < self.dynamic_obs:
                pygame.draw.line(self.map_pyg, (255, 255, 0), Grouped_Obs_center[_], Grouped_Obs_Vend[_], width=2)
        self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

        # draw path
        for i in range(len(self.waypoints)-1):
            pygame.draw.line(self.canvas, (0, 0, 255), self.waypoints[i], self.waypoints[i+1], width=2)

        # draw starting and end points
        pygame.draw.circle(self.canvas, (255, 0, 0), self.waypoints[0], 5)
        pygame.draw.circle(self.canvas, (0, 255, 0), self.waypoints[-1], 5)

        self.window.blit(self.canvas, self.map_pyg.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.FPS)


    def Dynamic_Plan(self):
        self._Map_Init()
        c, TPP, collide = 0, 0, False
        while True:
            '''Step 1: Prepare the Grid Map with pygame'''
            self.grid_map.fill((0,0,0))
            Grouped_Obs_Segments = self.Grouped_Obs_Segments.int().numpy()
            for _ in range(self.O):
                pygame.draw.polygon(self.grid_map, (1,1,1), Grouped_Obs_Segments[_, :, 0, :])
            map_data_np = (pygame.surfarray.array3d(self.grid_map)[:, :, 0]) #(366,366)
            map_data_np = scipy.ndimage.binary_dilation(map_data_np, structure=self.kernel)


            '''Step 2: Collide Detection'''
            if map_data_np[round(self.x_start), round(self.y_start)]:
                collide = True

            '''Step 3: Find path with Planner'''
            if not collide:
                env = Map(x_range=self.window_size, y_range=self.window_size, Obstacle_Segments=self.Obs_Segments, occupancy_grid_map=map_data_np)
                self.waypoints = None
                c += 1
                t0 = time.time()
                max_dist, sample_num = 15, 10000
                while self.waypoints is None:
                    max_dist += (np.random.random()-0.5)
                    sample_num += 100
                    if self.PlannerIdx==0: # RRT
                        planner = self.search_factory(self.planner_name, start=(self.x_start,self.y_start), goal=(self.x_target,self.y_target), env=env, max_dist=max_dist, sample_num=sample_num)
                    else: # RRT*
                        planner = self.search_factory(self.planner_name, start=(self.x_start,self.y_start), goal=(self.x_target,self.y_target), env=env, max_dist=max_dist, r=100, sample_num=sample_num)
                    cost, self.waypoints = planner.plan()
                    # if self.waypoints is None:
                    #     print(f'startpoint:{(self.x_start,self.y_start)}, targetpoint:{(self.x_target,self.y_target)}')
                    #     start_node = Node((self.x_start,self.y_start), None, 0, 0)
                    #     target_node = Node((self.x_target,self.y_target), None, 0, 0)
                    #     if planner.isInsideObs(start_node): print('start node is in obs')
                    #     if planner.isInsideObs(target_node): print('target node is in obs')
                    #     torch.save(self.Obs_Segments,'Generate_Obstacle_Segments/Obstacle_Segments.pt')
                if (self.x_target, self.y_target) != self.waypoints[-1]: self.waypoints.reverse()  # self.waypoints[0]
                TPP = TPP + ((time.time() - t0) - TPP) / c  # Time Per Planning

            '''Step 4: Update the map and Render'''
            if self.render: self._render_frame()
            self._update_map()


            '''Step 5: Stop Determination'''
            if self.d2target < self.reach_trsd:
                return self.travel_distance, TPP
            if collide:
                time.sleep(1)
                return 0, TPP #Collide

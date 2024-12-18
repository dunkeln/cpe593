import os, sys
import numpy as np
from itertools import product

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .local_planner import LocalPlanner
from utils import Env, Robot

class DWA(LocalPlanner):
    '''
    Class for Dynamic Window Approach(DWA) motion planning.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment
    heuristic_type: str
        heuristic function type, default is euclidean

    Examples
    ----------
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # kinematic parameters
        kinematic = {}
        kinematic["V_MAX"]         = 1.0;               #  maximum velocity [m/s]
        kinematic["W_MAX"]         = 30.0 * np.pi /180; #  maximum rotation speed[rad/s]
        kinematic["V_ACC"]         = 0.2;               #  acceleration [m/s^2]
        kinematic["W_ACC"]         = 50.0 * np.pi /180; #  angular acceleration [rad/s^2]
        kinematic["V_RESOLUTION"]  = 0.01;              #  velocity resolution [m/s]
        kinematic["W_RESOLUTION"]  = 1.0 * np.pi /180;  #  rotation speed resolution [rad/s]]
        # robot
        self.robot = Robot(start[0], start[1], start[2], 0, 0, **kinematic)
        # evalution parameters
        self.eval_param = {"heading": 0.045,
                           "distance": 0.1,
                           "velocity": 0.1,
                           "predict_time": 3.0,
                           "dt": 0.1,
                           "R": 2.0
                        }
        # threshold
        self.max_iter = 2000
        self.max_dist = 1.0

    def __str__(self) -> str:
        return "Dynamic Window Approach(DWA)"

    def plan(self):
        '''
        Dynamic Window Approach(DWA) motion plan function.
        [1] The Dynamic Window Approach to Collision Avoidance.
        '''
        history_traj = []
        for _ in range(self.max_iter):
            # dynamic configure
            vr = self.calDynamicWin()
            eval_win, traj_win = self.evaluation(vr)
        
            # failed
            if not len(eval_win):
                return
            
            # update
            max_index = np.argmax(eval_win[:, -1])
            u = np.expand_dims(eval_win[max_index, 0:-1], axis=1)
            self.robot.kinematic(u, self.eval_param["dt"])
            history_traj.append(traj_win)
            
            # goal found
            if self.dist((self.robot.px, self.robot.py), self.goal) < self.max_dist:
                return history_traj, self.robot.history_pose

        return history_traj, self.robot.history_pose

    def calDynamicWin(self) -> list:
        '''
        Calculate dynamic window.

        Return
        ----------
        v_reference: list
            reference velocity
        '''
        # hard margin
        vs = (0, self.robot.V_MAX, -self.robot.W_MAX, self.robot.W_MAX)
        # predict margin
        vd = (self.robot.v - self.robot.V_ACC * self.eval_param["dt"], 
              self.robot.v + self.robot.V_ACC * self.eval_param["dt"], 
              self.robot.w - self.robot.W_ACC * self.eval_param["dt"],
              self.robot.w + self.robot.W_ACC * self.eval_param["dt"]
            )

        # dynamic window
        v_tmp = np.array([vs, vd])
        # reference velocity
        vr = [float(np.max(v_tmp[:, 0])), float(np.min(v_tmp[:, 1])), 
                float(np.max(v_tmp[:, 2])), float(np.min(v_tmp[:, 3]))]
        return vr

    def evaluation(self, vr):
        '''
        Extract the path based on the CLOSED set.

        Parameters
        ----------
        closed_set: list
            CLOSED set

        Return
        ----------
        cost: float
            the cost of planning path
        path: list
            the planning path
        '''
        v_start, v_end, w_start, w_end = vr
        v = np.linspace(v_start, v_end, num=int((v_end - v_start) / self.robot.V_RESOLUTION)).tolist()
        w = np.linspace(w_start, w_end, num=int((w_end - w_start) / self.robot.W_RESOLUTION)).tolist()
        
        eval_win, traj_win = [], []
        for v_, w_ in product(v, w):
            # trajectory prediction, consistent of poses
            traj = self.generateTraj(v_, w_)
            end_state = traj[-1].squeeze().tolist()
            
            # heading evaluation
            theta = self.angle((end_state[0], end_state[1]), self.goal)
            heading = np.pi - abs(end_state[2] - theta)

            # obstacle evaluation
            dist_vector = np.array(tuple(self.obstacles)) - np.array([end_state[0], end_state[1]])
            dist_vector = np.sqrt(np.sum(dist_vector**2, axis=1))
            distance = np.min(dist_vector)
            if distance > self.eval_param["R"]:
                distance = self.eval_param["R"]

            # velocity evaluation
            velocity = abs(v_)
            
            # braking evaluation
            dist_stop = v_ * v_ / (2 * self.robot.V_ACC)

            # collision check
            if distance > dist_stop and distance >= 1:
                eval_win.append((v_, w_, heading, distance, velocity))
                traj_win.append(traj)
    
        # normalization
        eval_win = np.array(eval_win)
        if np.sum(eval_win[:, 2]) != 0:
            eval_win[:, 2] = eval_win[:, 2] / np.sum(eval_win[:, 2])
        if np.sum(eval_win[:, 3]) != 0:
            eval_win[:, 3] = eval_win[:, 3] / np.sum(eval_win[:, 3])
        if np.sum(eval_win[:, 4]) != 0:
            eval_win[:, 4] = eval_win[:, 4] / np.sum(eval_win[:, 4])

        # evaluation
        factor = np.array([[1, 0,                          0],
                           [0, 1,                          0],
                           [0, 0, self.eval_param["heading"]],
                           [0, 0, self.eval_param["distance"]],
                           [0, 0, self.eval_param["velocity"]]])

        return eval_win @ factor, traj_win


    def generateTraj(self, v, w):
        '''
        Generate predict trajectory.

        Return
        ----------
        v_reference: list
            reference velocity
        '''
        u = np.array([[v], [w]])
        state = self.robot.state
        time_steps = int(self.eval_param["predict_time"] / self.eval_param["dt"])
        
        traj = []
        for _ in range(time_steps):
            state = self.robot.lookforward(state, u, self.eval_param["dt"])
            traj.append(state)
        
        return traj

    def run(self):
        '''
        Running both plannig and animation.
        '''
        history_traj, history_pose = self.plan()

        if not history_pose:
            raise ValueError("Path not found!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        # for traj in history_traj[-1]:
            
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

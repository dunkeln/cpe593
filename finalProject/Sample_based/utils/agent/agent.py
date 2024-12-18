from abc import abstractmethod, ABC
import numpy as np


class Agent(ABC):
    '''
    Abstract class for agent.

    Parameters
    ----------
    px: float
        initial x-position
    py: float
        initial y-position
    theta: float
        initial pose angle
    '''    
    def __init__(self, px, py, theta) -> None:
        self.px = px
        self.py = py
        self.theta = theta

    @abstractmethod
    def kinematic(self, u, dt):
        pass

    @property
    @abstractmethod
    def state(self):
        pass


class Robot(Agent):
    '''
    Class for robot.

    Parameters
    ----------
    px: float
        initial x-position
    py: float
        initial y-position
    theta: float
        initial pose angle
    v: float
        linear velocity
    w: float
        angular velocity
    '''  
    def __init__(self, px, py, theta, v, w, **parameters) -> None:
        super().__init__(px, py, theta)
        # velocity
        self.v = v
        self.w = w
        # history
        self.history_pose = []
        # other parameters
        self.parameters = parameters
        for param, val in parameters.items():
            setattr(self, param, val)
    
    def __str__(self) -> str:
        return "Robot"
    
    def kinematic(self, u, dt, replace=True):
        new_state = self.lookforward(self.state, u, dt).squeeze().tolist()
        if replace:
            self.history_pose.append((self.px, self.py, self.theta))
            self.px, self.py, self.theta = new_state[0], new_state[1], new_state[2]
            self.v, self.w = new_state[3], new_state[4]
        else:
            return Robot(new_state[0], new_state[1], new_state[2], 
                new_state[3], new_state[4], **self.parameters)
    
    def lookforward(self, state, u, dt):
        F = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        B = np.array([[dt * np.cos(self.theta),  0],
                      [dt * np.sin(self.theta),  0],
                      [                      0, dt],
                      [                      1,  0],
                      [                      0,  1]])
        new_state = F @ state + B @ u
        return new_state

    def reset(self):
        self.v = 0
        self.w = 0
        self.history_pose = []

    @property
    def state(self):
        state = np.array([[self.px], [self.py], [self.theta], [self.v], [self.w]]) 
        return state

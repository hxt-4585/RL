import numpy as np

class GridWorld:
    def __init__(self, grid_size=(4, 4), goal_state=(3, 3), obstacles=None):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.obstacles = obstacles if obstacles is not None else []
        self.reset()
        
    
    def reset(self):
        self.agent_pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        while self.agent_pos == self.goal_state or self.agent_pos in self.obstacles:
            self.agent_pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        return self.agent_pos
    
    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # 上
            new_pos = (max(x - 1, 0), y)
        elif action == 1:  # 下
            new_pos = (min(x + 1, self.grid_size[0] - 1), y)
        elif action == 2:  # 左
            new_pos = (x, max(y - 1, 0))
        elif action == 3:  # 右
            new_pos = (x, min(y + 1, self.grid_size[1] - 1))

        if new_pos in self.obstacles:
            reward = -10 
        else:
            reward = -1
        
        self.agent_pos = new_pos
        
        if self.agent_pos == self.goal_state:
            return self.agent_pos, 0, True, {}
        else:
            return self.agent_pos, reward, False, {}
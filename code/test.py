import numpy as np
from collections import defaultdict

# 定义网格世界环境
class GridWorld:
    def __init__(self, grid_size=(4, 4), goal_state=(3, 3), obstacles=None):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.obstacles = obstacles if obstacles is not None else []
        self.reset()

    def reset(self):
        # 重置智能体位置到随机位置（不是目标位置或障碍物位置）
        self.agent_pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        while self.agent_pos == self.goal_state or self.agent_pos in self.obstacles:
            self.agent_pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        return self.agent_pos

    def step(self, action):
        # 根据动作更新位置
        x, y = self.agent_pos
        if action == 0:  # 上
            new_pos = (max(x - 1, 0), y)
        elif action == 1:  # 下
            new_pos = (min(x + 1, self.grid_size[0] - 1), y)
        elif action == 2:  # 左
            new_pos = (x, max(y - 1, 0))
        elif action == 3:  # 右
            new_pos = (x, min(y + 1, self.grid_size[1] - 1))

        # 检查新位置是否是障碍物
        if new_pos in self.obstacles:
            # 如果是障碍物，施加更大的惩罚
            reward = -10  # 设定惩罚力度
        else:
            reward = -1  # 普通步伐的惩罚
        
        self.agent_pos = new_pos
        
        # 计算奖励，是否结束
        if self.agent_pos == self.goal_state:
            return self.agent_pos, 0, True, {}
        else:
            return self.agent_pos, reward, False, {}

# 定义策略评估和改进过程
def mc_exploring_starts(env, num_episodes, gamma=0.1):
    Q = defaultdict(lambda: np.zeros(4))  # 动作价值函数，Q(s, a)
    returns = defaultdict(list)  # 用于存储每个状态-动作对的回报
    
    for episode_num in range(num_episodes):
        # 1. 随机选择一个起始状态和动作
        state = env.reset()  # 重置环境并随机初始化状态
        action = np.random.choice(4)  # 随机选择动作

        # 2. 生成一个episode
        episode = []
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))  # 记录当前状态-动作对和奖励
            state = next_state
            if not done:
                action = np.random.choice(4)  # 继续随机选择下一个动作

        # 3. 计算每个状态-动作对的累积回报
        G = 0
        episode.reverse()  # 从终点往回计算回报
        for (state, action, reward) in episode:
            G = reward + gamma * G  # 累积回报
            # 如果该状态-动作对是第一次在episode中出现
            if (state, action) not in [(x[0], x[1]) for x in episode[:-1]]:
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])  # 更新Q值

    return Q

# 设置障碍物位置
obstacles = [(1, 1), (1, 2), (2, 1)]  # 在网格中定义障碍物

# 创建网格环境并运行算法
env = GridWorld(obstacles=obstacles)
Q = mc_exploring_starts(env, num_episodes=1000)

# 打印学习到的Q值
for state in Q:
    print(f"State {state}: {Q[state]}")

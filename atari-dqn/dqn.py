import torch
import numpy as np
from agent import Network, Agent
from consts import *
from replaybuffer import ReplayBuffer



class DQN(Network):
    def __init__(self, state_size, action_size, width, depth):
        self.state_size = state_size
        self.action_size = action_size

        layers = [
            torch.nn.Linear(self.state_size, width, dtype=torch.float32),
            torch.nn.ReLU()
        ]

        for _ in range(depth):
            layers.extend([
                torch.nn.Linear(width, width, dtype=torch.float32),
                torch.nn.ReLU()
            ])

        layers.extend([
            torch.nn.Linear(width, self.action_size, dtype=torch.float32)
        ])

        super.__init__()

        self.neuralnet = torch.nn.Sequential(layers)
    
    def forward(self, x):
        x = x.type(torch.float32)
        return self.neuralnet(x)



class DQNAgent(Agent):
    def __init__(self, state_size, action_size, width, depth, gamma, batch_size, learn_rate):
        super().__init__(state_size, action_size)

        self.current = DQN(self.state_size, self.action_size, width, depth).to(DEVICE)
        self.target = DQN(self.state_size, self.action_size, width, depth).to(DEVICE)

        self.gamma = gamma

        for p in self.target.parameters():
            p.requires_grad = False

        self.replay_buffer = self.init_replay_buffer()
        self.batch_size = batch_size

        self.learn_rate = learn_rate
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.learn_rate)
    
    def init_replay_buffer(self):
        replay_buffer = ReplayBuffer(1e5, obs_dtype=np.float32, act_dtype=np.float32, default_dtype=np.float32)
        return replay_buffer
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
    
    def sample(self):
        pass
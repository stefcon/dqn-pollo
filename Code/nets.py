import torch
import torch.nn as nn

from consts import *


class DQN(nn.Module):
    def __init__(self, s_size, a_size, h=16, num_hidden=1):
        self.state_size = s_size
        self.action_size = a_size

        modules = [
            nn.Linear(self.state_size, h, dtype=torch.float32),
            nn.ReLU()
        ]

        for _ in range(num_hidden):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(h, self.action_size, dtype=torch.float32))

        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(*modules).to(DEVICE)

    def forward(self, x):
        x = x.type(torch.float32)
        return self.neuralnet(x)


class DuelingDQN(nn.Module):
    def __init__(self, s_size, a_size, h=16, num_hidden=1):
        super(DuelingDQN, self).__init__()
        self.state_size = s_size
        self.action_size = a_size

        modules = [
            nn.Linear(self.state_size, h, dtype=torch.float32),
            nn.ReLU()
        ]

        for _ in range(num_hidden-1):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            modules.append(nn.ReLU())

        self.neuralnet = nn.Sequential(*modules).to(DEVICE)

        # Adding two new heads: advantage and value networks
        advantage_modules = [self.neuralnet, nn.Linear(h,h, dtype=torch.float32), nn.ReLU(), 
                            nn.Linear(h, self.action_size)]
        value_modules = [self.neuralnet, nn.Linear(h,h, dtype=torch.float32), nn.ReLU(), 
                            nn.Linear(h, self.action_size)]

        self.advantage_net = nn.Sequential(*advantage_modules).to(DEVICE)
        self.value_net = nn.Sequential(*value_modules).to(DEVICE)
        

    def forward(self, x):
        x = x.type(torch.float32)
        advantage = self.advantage_net(x)
        advantage -= advantage.mean()
        value = self.value_net(x)
        q = advantage + value
        return q


        



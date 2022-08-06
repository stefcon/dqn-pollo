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
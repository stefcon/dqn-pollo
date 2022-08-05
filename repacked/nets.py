import torch
import torch.nn as nn

from consts import WINDOW_HEIGHT, WINDOW_WIDTH

from torchsummary import summary


class DQN(nn.Module):
    def __init__(self, s_size, a_size, h=16, num_hidden=1):
        self.state_size = s_size
        self.action_size = a_size

        # 160 > 156 > 152 >> 76 > 72 >> 36 > 32 >> 16 > 12 >> 6 > 2 >> 1

        modules = [
            nn.Conv2d(3, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Flatten(),
            nn.Linear(384, h, dtype=torch.float32),
            nn.ReLU()
        ]

        for _ in range(num_hidden):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(h, self.action_size, dtype=torch.float32))

        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(*modules)

        print("IS IT")
        summary(self.neuralnet, input_size=(3, 210, 160))
        print("OKAY")

    def forward(self, x):
        x = x.type(torch.float32).view((-1, 3, WINDOW_HEIGHT, WINDOW_WIDTH))
        return self.neuralnet(x)
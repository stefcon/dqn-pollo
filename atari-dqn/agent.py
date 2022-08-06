import torch


class Network(torch.nn.Module):
    def name(self):
        return "NETWORK"


class Agent:
    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
    
    def remember(self, state, action, reward, next_state, done):
        pass

    def sample(self):
        pass

    def select_action(self, state):
        pass

    def backward(self):
        pass

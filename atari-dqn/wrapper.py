from typing import Any
import gym

class Wrapper:
    def __init__(self, gym_env: gym.Env) -> None:
        self.env = gym_env
    
    def state_size(self):
        return self.env.observation_space.shape[0]
    
    def action_size(self):
        return self.env.action_space.shape[0]
    
    def render(self):
        return self.env.render()
    
    def reset(self):
        return self.env.reset()

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        print(terminated, truncated, info)
        return observation, reward, terminated or truncated


import gym

from consts import *

def initialize():
    print("INITIALIZING!")
    env = gym.make(ENV_NAME, new_step_api=True)


def train():
    initialize()
    print("TRAINING!")

if __name__ == "__main__":
    train()

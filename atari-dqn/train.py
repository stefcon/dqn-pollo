import gym


def initialize():
    print("INITIALIZING!")
    env = gym.make("Pong-v4", new_step_api=True)


def train():
    initialize()
    print("TRAINING!")

if __name__ == "__main__":
    train()

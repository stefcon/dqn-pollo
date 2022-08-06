import time
import random
import os
from utils import create_run_name, visualize_result
from dqn import DQNAgent
from env_wrapper import EnvWrapper
from consts import *

import argparse

import torch
import gym


def play(model, visualize=True):
    env = EnvWrapper(gym_env=gym.make(ENV_NAME, render_mode='human'), steps=STEPS)
    agent = DQNAgent(
        state_size=env.state_size(),
        action_size=env.action_size(),
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_hidden=NUM_H,
        hidden_units=H
    )

    agent.current.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    
    state = env.reset()
    total_reward = 0.0

    for _ in range(STEPS):
        action = agent.select_action(torch.from_numpy(state).to(DEVICE).detach())
        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            break
        state = next_state


if __name__ == '__main__':
    model = None
    visualize = True

    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--model', type=str, required=True)
    parser.add_argument('-v','--visualize', type=bool, required=False)
    # Parse the argument
    args = parser.parse_args()
    
    model = args.model
    if args.visualize is not None:
        visualize = args.visualize

    play(model, visualize)
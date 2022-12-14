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


def play(model, episodes = 10, render=True):
    if render:
        env = EnvWrapper(gym_env=gym.make(ENV_NAME, new_step_api=True, render_mode='human'), steps=STEPS, film_video=False)
    else:
        env = EnvWrapper(gym_env=gym.make(ENV_NAME, new_step_api=True), steps=STEPS, film_video=False)
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
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(STEPS):
            action = agent.select_action(torch.from_numpy(state).unsqueeze(0).to(DEVICE).detach()).item()
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                print("EPISODE: {0: <4}/{1: >4} | SCORE: {2: <7.1f}".format(episode + 1, episodes, total_reward))
                break
            state = next_state


if __name__ == '__main__':
    model = None
    episodes = 10
    render = False
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--model', type=str, required=True)
    parser.add_argument('-e','--episodes', type=int, required=False)
    parser.add_argument('-r','--render', type=bool, required=False)
    # Parse the argument
    args = parser.parse_args()
    
    model = args.model

    if args.episodes:
        episodes = args.episodes
    if args.render:
        render = args.render
    
    play(model, episodes, render)
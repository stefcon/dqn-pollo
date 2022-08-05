from pydoc import render_doc
import time
import random
import os
from utils import create_run_name, visualize_result
from dqn import DQNAgent
from env_wrapper import EnvWrapper
from consts import *

import torch
import gym


run_name = create_run_name(
        alg='DQN',
        env=ENV_NAME,
        num_layers=NUM_H,
        hidden_dim=H,
        eps_start=EPSILON_START,
        eps_end=EPSILON_END,
        decay=EPSILON_DECAY,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_ep=EPISODES,
        num_step=STEPS,
        updt_freq=UPDATE_FREQ,
        sw_freq=TARGET_FREQ
    )

def train():
    env = EnvWrapper(gym_env=gym.make(ENV_NAME, render_mode='human'), steps=STEPS)
    # Initialize Q networks, replay memory
    agent = DQNAgent(
        state_size=env.state_size(),
        action_size=env.action_size(),
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_hidden=NUM_H,
        hidden_units=H
    )

    epsilon = EPSILON_START
    results = []
    best_cum_rew = -1 # We will store only the best model (DQN can have "catastrophic forgetting")
    best_network = None # State dict of the best network
    td_errors = []
    start = time.time()
    random.seed(0)

    for episode in range(EPISODES):
        # Start game/episode
        state = env.reset()
        cum_rew = 0

        if episode > 10 and episode % TARGET_FREQ == 0:
            agent.update_target_model()

        # Loop inside one game episode
        for t in range(STEPS):
            
            # Display the game. Comment bellow line in order to get faster training.
            if episode > 140:
                env.render()
            
            if random.random() <= epsilon:
                action = env.env.action_space.sample()
            else:
                # Action selection
                action = agent.select_action(state=torch.from_numpy(state).to(DEVICE).detach())
                

            next_state, reward, done = env.step(action)
            cum_rew += reward
            
            # Store transition to replay memory
            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=float(done))
            
            # Update Q-values
            if episode > 10 and (episode + t) % UPDATE_FREQ == 0:
                # Q-value update
                td_error = agent.backward()
                td_errors.append(td_error)

            if done or (t == STEPS - 1):
                if episode > 10:
                    print("EPISODE: {0: <4}/{1: >4} | EXPLORE RATE: {2: <7.4f} | SCORE: {3: <7.1f}"
                        " | TD ERROR: {4: <5.2f} ".format(episode + 1, EPISODES, epsilon, cum_rew, td_error))
                else:
                    print("EPISODE: {0: <4}/{1: >4} | EXPLORE RATE: {2: <7.4f} | SCORE: {3: <7.1f}"
                        " | WARMUP - NO TD ERROR".format(episode + 1, EPISODES, epsilon, cum_rew))
                results.append(cum_rew)
                if best_cum_rew is None or best_cum_rew <= cum_rew:
                            best_network = agent.current.state_dict()
                            best_cum_rew = cum_rew
                            if best_cum_rew is not None:
                                print(f'Best mean reward updated {best_cum_rew:.3f}')
                if done:
                    break

            state = next_state

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

    end = time.time()
    total_time = end - start

    # Saving parameters for model with biggest cumulative reward
    torch.save(best_network, run_name + '-best.dat')

    print()
    print("TOTAL EPISODES: {0: <4} | TOTAL UPDATE STEPS: {1: <7} | TOTAL TIME [s]: {2: <7.2f}"
        .format(EPISODES, len(td_errors), total_time))
    print("EP PER SECOND: {0: >10.6f}".format(total_time / EPISODES))
    print("STEP PER SECOND: {0: >8.6f}".format(total_time / len(td_errors)))

    fig = visualize_result(
        returns=results,
        td_errors=td_errors,
        policy_errors=None
    )
    fig.savefig(os.path.join(STORE_PATH, run_name + '.png'), dpi=400)


if __name__ == "__main__":
    train()
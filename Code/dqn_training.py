from cProfile import run
from cmath import inf
from pydoc import render_doc
import time
import random
import os
from utils import create_run_name, visualize_result, moving_average
from dqn import DQNAgent
from env_wrapper import EnvWrapper
from consts import *

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gym

from multiprocessing import Pool

run_name = create_run_name(
        alg='DQN',
        env='lander',
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
        sw_freq=TARGET_FREQ,
        is_double=DOUBLE
    )

def train(run_id, t_gamma, t_lr, t_eps_start, t_eps_end, t_num_hidden, t_hidden_units):

    print ("[{}] INITIALIZING [G={},LR={},ES={},EE={},NH={},H={}]".format(run_id, t_gamma, t_lr, t_eps_start, t_eps_end, t_num_hidden, t_hidden_units))
    env = EnvWrapper(gym_env=gym.make(ENV_NAME, new_step_api=True), steps=STEPS)
    # Initialize Q networks, replay memory
    agent = DQNAgent(
        state_size=env.state_size(),
        action_size=env.action_size(),
        gamma=t_gamma,
        batch_size=BATCH_SIZE,
        lr=t_lr,
        num_hidden=t_num_hidden,
        hidden_units=t_hidden_units,
        eps_start=t_eps_start,
        eps_end=t_eps_end,
        decay=EPSILON_DECAY,
        exp_decay=EXP_DECAY,
        is_double=DOUBLE
    )

    summary_writer = SummaryWriter()

    epsilon = EPSILON_START
    results = []
    best_mean_rew = None # We will store only the best model (DQN can have "catastrophic forgetting")
    best_network = None # State dict of the best network

    td_errors = []
    td_error = 0

    start = time.time()
    decay_step = 0
    random.seed(0)

    for episode in range(EPISODES):
        # Start game/episode
        state = env.reset()
        cum_rew = 0

        if episode > WARMUP and episode % TARGET_FREQ == 0:
            agent.update_target_model()

        # Loop inside one game episode
        for t in range(STEPS):
            decay_step += 1
            
            # Display the game. Comment bellow line in order to get faster training.
            # if episode > WARMUP:
            #     env.render()
            
            # Choose next action for the agent to take
            # Unsqueezing since select_action expects dim=2 tensors!
            action = agent.eps_action(env, state, epsilon)
                
            # Execute action in the environment and save the reward
            next_state, reward, done = env.step(action)
            cum_rew += reward
            
            # Store transition to replay memory
            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=float(done))
            
            # Update Q-values
            if episode > WARMUP and (episode*STEPS + t) % UPDATE_FREQ == 0:
                # Q-value update
                td_error = agent.backward()
                td_errors.append(td_error)

            if done or (t == STEPS - 1):
                bmr = best_mean_rew
                if (best_mean_rew == None):
                    bmr = -inf
                if episode > WARMUP:
                    print("[{0}] EPISODE: {1: >4}/{2: >4} | EXPRATE: {3: >7.4f} | SCORE: {4: >7.1f} | TD ERROR: {5: >5.2f} | BMR {6: >9.3f}".format(run_id, episode + 1, EPISODES, epsilon, cum_rew, td_error, bmr))
                else:
                    print("[{0}] EPISODE: {1: >4}/{2: >4} | EXPRATE: {3: >7.4f} | SCORE: {4: >7.1f} | WRMP NO TD ERR | BMR {5: >9.3f}".format(run_id, episode + 1, EPISODES, epsilon, cum_rew, bmr))

                # Live graphs in TensorBoard
                summary_writer.add_scalar("Score", cum_rew, episode)
                summary_writer.add_scalar("Explore rate", epsilon, episode)
                summary_writer.add_scalar("TD Error", td_error, episode)

                results.append(cum_rew)

                # Save the best model
                if episode > ROLLING_PERIOD:
                    mean_reward = moving_average(results[-ROLLING_PERIOD:], ROLLING_PERIOD)[0]
                    if best_mean_rew is None or best_mean_rew < mean_reward:
                                best_network = agent.current.state_dict()
                                best_mean_rew = mean_reward
                                # if best_mean_rew is not None:
                                #     print(f'Best mean reward updated {best_mean_rew:.3f}')
                if done:
                    break
            
            # Transitions to agent's next state
            state = next_state

        # Execute epsilon decay for agents exploration policy (at the end of an episode)
        if episode > WARMUP:
            epsilon = agent.epsilon_decay(epsilon, decay_step)

    end = time.time()
    total_time = end - start

    # Saving parameters for model with biggest cumulative reward
    torch.save(best_network, os.path.join(BEST_MODELS, "[" + run_id + "]" + run_name + '-best.dat'))

    print()
    print("[{0}] TOTAL EPISODES: {1: <4} | TOTAL UPDATE STEPS: {2: <7} | TOTAL TIME [s]: {3: <7.2f}".format(run_id, EPISODES, len(td_errors), total_time))
    print("[{0}] EP PER SECOND: {1: >10.6f}".format(run_id, total_time / EPISODES))
    print("[{0}] STEP PER SECOND: {1: >8.6f}".format(run_id, total_time / len(td_errors)))

    fig = visualize_result(
        returns=results,
        td_errors=td_errors,
        policy_errors=None
    )
    fig.savefig(os.path.join(GRAPH_PATH, "[" + run_id + "]" + run_name + '.png'), dpi=400)

    return run_id


def yoink(x, k):
    print("ENTERING FOR", x, "-", k)

    for i in range(x):
        print("  #:", i, "/", x)

    print("LEAVING FOR", x, "-", k)

    return x


if __name__ == "__main__":
    args = []

    for lrid, lr in [ ("L15", 0.0015), ("L10", 0.001), ("L05", 0.0005) ]:
        for esid, eps_start in [ ("E5", 0.5), ("E7", 0.75) ]:
            for gmid, gamma in [ ("G1", 0.99), ("G2", 0.995) ]:
                args.append( (lrid + "-" + esid + "-" + gmid, gamma, lr, eps_start, EPSILON_END, NUM_H, H) )

    with Pool() as pool:
            # Experimenting with learning rate
        # LR = lr
        # train()
        res = [pool.apply_async(train, arg) for arg in args]

        print("ASDF")

        res = [r.get() for r in res]
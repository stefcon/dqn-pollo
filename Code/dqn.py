import os
import random
import torch
import time
import numpy as np
from consts import *

from nets import DQN
from rb import ReplayBuffer


class DQNAgent(object):
    def __init__(self, state_size, action_size, gamma=0.95, batch_size=256, lr=0.00025, num_hidden=2,
                 hidden_units=64, exp_decay=True, is_double=False):
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.exp_decay = exp_decay
        self.is_double = is_double
        self.name = 'DQN'

        # We create "live" and "target" networks from the original paper.
        self.current = DQN(state_size, action_size, h=hidden_units, num_hidden=num_hidden).to(DEVICE)
        self.target = DQN(state_size, action_size, h=hidden_units, num_hidden=num_hidden).to(DEVICE)
        for p in self.target.parameters():
            p.requires_grad = False
        self.update_target_model()

        # Replay buffer (memory) initialization.
        self.rb = self.init_rb()
        self.batch_size = batch_size

        # Learning rate and optimizer used to update the "live" network in DQN.
        learning_rate = lr
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def init_rb(self):
        # Replay buffer initialization.
        replay_buffer = ReplayBuffer(1e5, obs_dtype=np.float32, act_dtype=np.int64, default_dtype=np.float32)
        return replay_buffer

    def remember(self, state, action, reward, next_state, done):
        # Remember (Q,S,A,R,S') as well as whether S' is terminating state.
        self.rb.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

    def sample(self):
        states, actions, next_states, rewards, dones = self.rb.sample(self.batch_size)
        one_hot_acts = torch.squeeze(
            torch.nn.functional.one_hot(actions, num_classes=self.action_size))
        return states, one_hot_acts, rewards, next_states, dones

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def select_action(self, state):
        # Getting "greedy" action
        action = torch.argmax(self.current(state)).item()
        return action

    def backward(self):
        # Sample mini-batch of stored transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.sample()

        # Gathering Q-values by looking what actions were taken in the past with current network
        qs_selected = torch.sum(self.current(states)*actions, dim=1)

        # Calculating target value with the "stale" network
        with torch.no_grad():
            if self.is_double:
                # Double DQN
                next_best_actions = self.select_action(next_states)
                next_best_one_hot = torch.squeeze(
                    torch.nn.functional.one_hot(next_best_actions, num_classes=self.action_size))
                q_target_ = torch.sum(self.target(next_states)*next_best_one_hot, dim=1)
            else:    
                # Vanilla DQN
                q_values = self.target(next_states)
                q_target_ = torch.max(q_values, dim=1)[0]

        qs_target = torch.squeeze(rewards) + (1 - torch.squeeze(dones))*self.gamma * q_target_

        # We calculate the absolute difference between current and target values q values,
        # which is useful info for debugging.
        with torch.no_grad():
            td_error = torch.abs(qs_target - qs_selected)

        # We update the "live" network, self.current. First we zero out the optimizer gradients
        # and then we apply the update step using qs_selected and qs_target.
        self.optimizer.zero_grad()
        loss = (torch.nn.functional.mse_loss(qs_selected, qs_target)).mean()
        loss.backward()
        # Limiting gradient step by clipping
        torch.nn.utils.clip_grad_norm(self.current.parameters(), 1)
        self.optimizer.step()
        return torch.mean(td_error).item()
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
                 hidden_units=64, eps_start=1, eps_end=0.2, decay=0.05, exp_decay=False, is_double=False):
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
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
        return states, actions, rewards, next_states, dones

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def select_action(self, state):
        """
        Getting "greedy" action
        """
        # We add a dimension, so that it works for batches out of the box
        action = torch.argmax(self.current(state), dim=1)
        return action

    def eps_action(self, env, state, epsilon):
        """
        Choosing epsilon-greedy action
        """
        if random.random() <= epsilon:
            return env.env.action_space.sample()
        # Action selection
        return self.select_action(state=torch.from_numpy(state).unsqueeze(0).to(DEVICE).detach()).item()

    def epsilon_decay(self, curr_eps, decay_step):
        if self.exp_decay:
            # Exponential epsilon decay:
            return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.decay * decay_step)
        else:
            # Linear decay
            if curr_eps > EPSILON_END:
                curr_eps = max(curr_eps * (1-EPSILON_DECAY), EPSILON_END)
            return curr_eps

    def backward(self):
        # Sample mini-batch of stored transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.sample()

        # Gathering Q-values by looking what actions were taken in the past with current network
        qs_selected = self.current(states).gather(1, actions).squeeze()

        # Calculating target value with the "stale" network
        with torch.no_grad():
            if self.is_double:
                # Double DQN
                next_best_actions = self.select_action(next_states)
                q_target_ = self.target(next_states).gather(1, next_best_actions.unsqueeze(1))
            else:    
                # Vanilla DQN
                q_values = self.target(next_states)
                q_target_ = torch.max(q_values, dim=1)[0]

        qs_target = rewards.squeeze() + (1 - dones.squeeze()) * self.gamma * q_target_.squeeze()

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
        torch.nn.utils.clip_grad_norm_(self.current.parameters(), 1)
        self.optimizer.step()
        return torch.mean(td_error).item()
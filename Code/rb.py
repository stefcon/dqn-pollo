import random
import numpy as np
import torch
from consts import *


class ReplayBuffer(object):
    def __init__(self, storage_size, obs_dtype, act_dtype, default_dtype):
        self._data = []
        self._max_len = int(storage_size)

        self._o_dtype = obs_dtype
        self._a_dtype = act_dtype
        self._def_dtype = default_dtype
        self._next_id = 0
        self._write_flg = True
        np.random.seed(123)

    @staticmethod
    def _preprocess(var, dtype):
        if torch.is_tensor(var):
            var = var.detach().numpy()
        if hasattr(var, 'shape'):
            var = var.astype(dtype)
            var = np.squeeze(var)
        else:
            var = dtype(var)

        return var

    def add(self, obs, act, next_obs, rew, done):

        obs = self._preprocess(var=obs, dtype=self._o_dtype)
        act = self._preprocess(var=act, dtype=self._a_dtype)
        next_obs = self._preprocess(var=next_obs, dtype=self._o_dtype)
        rew = self._preprocess(var=rew, dtype=self._def_dtype)
        done = self._preprocess(var=done, dtype=self._def_dtype)

        if len(self._data) < self._max_len:
            self._data.append((obs, act, next_obs, rew, done))
        else:
            self._data[self._next_id] = (obs, act, next_obs, rew, done)

        self._next_id = (self._next_id + 1) % self._max_len

    def sample(self, batch_size):
        cap = len(self._data)
        if cap < batch_size:
            replace = True
        else:
            replace = False
        idxs = list(np.random.choice(cap, batch_size, replace=replace))
        obss, actions, next_obss, rewards, dones = [], [], [], [], []

        for idx in idxs:
            data = self._data[idx]
            obss.append(data[0])
            actions.append(data[1])
            next_obss.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])

            obss = np.asarray(obss).reshape((batch_size, -1))
            actions = np.asarray(actions).reshape((batch_size, -1))
            next_obss = np.asarray(next_obss).reshape((batch_size, -1))
            rewards = np.asarray(rewards).reshape((batch_size, -1))
            dones = np.asarray(dones).reshape((batch_size, -1))

            return torch.from_numpy(obss).to(DEVICE), torch.from_numpy(actions).to(DEVICE), \
            torch.from_numpy(next_obss).to(DEVICE), torch.from_numpy(rewards).to(DEVICE), \
            torch.from_numpy(dones).to(DEVICE)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.number = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        
        if idx >= self.capacity - 1:
            return idx
        
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        self.number = min(self.number + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def num(self):
        return self.number


class PERMemory(ReplayBuffer):
    def __init__(self, storage_size, obs_dtype, act_dtype, default_dtype, 
                alpha = 0.8, beta = 0.8, eps = 1e-2):
        super().__init__(storage_size, obs_dtype, act_dtype, default_dtype)
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.beta_multiplier_per_sampling = 1.0005
        self._data = SumTree()


    def add(self, obs, act, next_obs, rew, done):

        obs = self._preprocess(var=obs, dtype=self._o_dtype)
        act = self._preprocess(var=act, dtype=self._a_dtype)
        next_obs = self._preprocess(var=next_obs, dtype=self._o_dtype)
        rew = self._preprocess(var=rew, dtype=self._def_dtype)
        done = self._preprocess(var=done, dtype=self._def_dtype)

        # here use reward for initial p, instead of maximum for initial p
        p = rew
        self._data.add([obs, act, next_obs, rew, done], p)

    def update(self, batch_idx, batch_td_error):
        for idx, error in zip(batch_idx, batch_td_error):
            p = (error + self.eps)  ** self.alpha 
            self._data.update(idx, p)
        
    def num(self):
        return self._data.num()
    
    def sample(self, batch_size):
        
        # data_batch = []
        obss, actions, next_obss, rewards, dones = [], [], [], [], []
        idx_batch = []
        p_batch = []
        
        segment = self._data.total() / batch_size
        #print(self.mem.total())
        #print(segment * batch_size)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print(s < self.mem.total())
            idx, p, data = self._data.get(s)

            # data_batch.append(data)
            obss.append(data[0])
            actions.append(data[1])
            next_obss.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])

            idx_batch.append(idx)
            p_batch.append(p)
        
        p_batch = (1.0/ np.array(p_batch) / self.mem_size) ** self.beta
        p_batch /= max(p_batch)
        
        self.beta = min(self.beta * self.beta_multiplier_per_sampling, 1)

        # obss = np.asarray([data[0] for data in data_batch]).reshape((batch_size, -1))
        # actions = np.asarray([data[1] for data in data_batch]).reshape((batch_size, -1))
        # next_obss = np.asarray([data[2] for data in data_batch]).reshape((batch_size, -1))
        # rewards = np.asarray([data[3] for data in data_batch]).reshape((batch_size, -1))
        # dones = np.asarray([data[4] for data in data_batch]).reshape((batch_size, -1))

        obss = np.asarray(obss).reshape((batch_size, -1))
        actions = np.asarray(actions).reshape((batch_size, -1))
        next_obss = np.asarray(next_obss).reshape((batch_size, -1))
        rewards = np.asarray(rewards).reshape((batch_size, -1))
        dones = np.asarray(dones).reshape((batch_size, -1))

        return torch.from_numpy(obss).to(DEVICE), torch.from_numpy(actions).to(DEVICE), \
         torch.from_numpy(next_obss).to(DEVICE), torch.from_numpy(rewards).to(DEVICE), \
         torch.from_numpy(dones).to(DEVICE), idx_batch, p_batch
    
        # return (data_batch, idx_batch, p_batch)
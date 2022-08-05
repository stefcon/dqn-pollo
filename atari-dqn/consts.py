import torch


# ENVIRONMENT
# ===========

# Path for storing training logs
STORE_PATH = '../logs/tmp_atari_dqn'
# Frames per second for rendering what's happening on screen
FPS = 25
# Torch device (CUDA if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# EPISODE
# =======

# Number of episodes to train for
EPISODES = 500
# Number of steps per episode
STEPS = 10000


# TRAINING
# ========

# Initial exploration rate
EPSILON_START = 1
# Final exploration rate
EPSILON_END = 0.001
# Exploration rate decay per episode
EPSILON_DECAY = 0.95
# Discount rate
GAMMA = 0.95
# DQN target network update frequency in episodes
TARGET_FREQ = 4
# DQN current network update frequency in steps
UPDATE_FREQ = 1
# Learning rate
LR = 0.00015
# Batch size
BATCH_SIZE = 256
# Environment name
ENV_NAME = 'Pong-v4'


# NETWORK ARCHITECTURE
# ====================

# Number of hidden layers of H units
NUM_H = 2
# Number of units in hidden layers
H = 64
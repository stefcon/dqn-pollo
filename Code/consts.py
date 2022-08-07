import torch

# Path for storing training info (should have better organisation)
GRAPH_PATH = './graphs'

# ### EPISODE ###################
# Number of episodes to train for
EPISODES = 650
# Number of steps per episode
STEPS = 1000

# Torch device (CUDA if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### TRAINING ##################
# Start exploration rate
# for eps greedy policy
EPSILON_START = 1.0
# End exploration rate
EPSILON_END = 0.01 # 0.001
# Exploration rate decay per EPISODE
EPSILON_DECAY =  0.005 #0.00002
# Exponential decay indicator
EXP_DECAY = False
# Discount rate
GAMMA = 0.999
# DQN Target network update
# frequency in EPISODES
TARGET_FREQ = 15
# DQN live network update
# frequency in STEPS
UPDATE_FREQ = 1
# Learning rate
LR = 0.001 # 0.00015
# Batch size
BATCH_SIZE = 128 #256
# Environment name
ENV_NAME = 'LunarLander-v2'
# ENV_NAME = 'CartPole-v1'
# Folder name for storing best models
BEST_MODELS = './best_models'
# Warmup period
WARMUP = 100
# Period for calculating mean rolling average
ROLLING_PERIOD = 50
# Using Double Q-Learning
DOUBLE = True

# ### NETWORK ARCHITECTURE ######
# Number of hidden layers
# of H units
NUM_H = 2
# Number of units in hidden
# layers
H = 64
# ### HYPER PARAMETERS END #######

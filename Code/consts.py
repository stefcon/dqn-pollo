import torch

# Path for storing training info (should have better organisation)
STORE_PATH = './tmp_dqn_learning'

# Frames per second for rendering what's happening on screen
FPS = 25

# ### EPISODE ###################
# Number of episodes to train for
EPISODES = 1000
# Number of steps per episode
STEPS = 100000

# Torch device (CUDA if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### TRAINING ##################
# Start exploration rate
# for eps greedy policy
EPSILON_START = 1.0
# End exploration rate
EPSILON_END = 0.02 # 0.001
# Exploration rate decay per EPISODE
EPSILON_DECAY = 0.00002 # 0.05
# Exponential decay indicator
EXP_DECAY = True
# Discount rate
GAMMA = 0.98
# DQN Target network update
# frequency in EPISODES
TARGET_FREQ = 4
# DQN live network update
# frequency in STEPS
UPDATE_FREQ = 1
# Learning rate
LR = 0.00005 # 0.00015
# Batch size
BATCH_SIZE = 64 #256
# Environment name
ENV_NAME = 'ALE/Pong-ram-v5'
# Folder name for storing best models
BEST_MODELS = 'best_models'

# ### NETWORK ARCHITECTURE ######
# Number of hidden layers
# of H units
NUM_H = 2
# Number of units in hidden
# layers
H = 64
# ### HYPER PARAMETERS END #######
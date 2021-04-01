# %%
from plot_to_gif import GifWrapper
import gym
import ma_gym
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from value_decomp_1 import *
# %matplotlib inline

from pathlib import Path

import pickle


# %%
ENV = "Switch4-v0"
num_agents = 4
env = gym.make(ENV, max_steps=50) # Use "Switch4-v0" for the Switch-4 game
env = GifWrapper(env)
agents_path = Path("agents4.pkl")
agents = load_agents(agents_path)

env.reset()
for a in agents:
    a.explore = False
    
 # %%
 eps_reward, num_done_test = test(agents, env, show=True)
# %%
eps_reward

# %%
num_done_test

# %%
agents[0].train_iter

# %%
agents[0].train_iter

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

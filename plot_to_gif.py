import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import time



import gym
class GifWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.ims = []

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        im = plt.imshow(self.env.render(mode='rgb_array'))
        self.ims.append([im])
        return next_state, reward, done, info

    def show(self):
        self.fig = plt.figure()
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, blit=True,repeat_delay=500, repeat=False)
        plt.show()
        return ani

# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# %%
import wandb
import gym
import pandas as pd
import ma_gym
from ma_gym.wrappers import Monitor
import matplotlib.pyplot as plt
import glob
import io
import base64
import random
from tqdm import tqdm, trange

from functools import partial
import torch.nn.functional as F
from collections import namedtuple
import torch.nn as nn
from torch import optim
import numpy as np
import torch
from IPython import display


# %%
wandb.init(project="switch4")

# %%
PASSIVE_AGENT = None
DEV = "cpu"

# %%
NUM_OBS = 5
DUELLING=True
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 100000
EPS = [0.2]
GAMMA = 0.9
LR = 0.001
CAPACITY = 3000
OPTIM_ITER = 10
TARGET_UPDATE_ITER = 20
BATCH_SIZE = 128
HIDDEN_DIM = 45
NUM_OPTIM_UPDATES = 30
TAU = 0.005
EPISODES = 10000




# %%
wandb.config.arch = "DUELLING_fingerprint_sin" if DUELLING else "DQnet"
wandb.config.passive_agent = PASSIVE_AGENT
wandb.config.eps = EPS
wandb.config.eps_start = EPS_START
wandb.config.eps_end = EPS_END
wandb.config.eps_decay = EPS_DECAY
wandb.config.gamma = GAMMA
wandb.config.target_update_iter = TARGET_UPDATE_ITER
wandb.config.replay_capacity = CAPACITY
wandb.config.lr = LR
wandb.config.batch_size = BATCH_SIZE
wandb.config.hidden_dim = HIDDEN_DIM
wandb.config.num_updates_per_e = NUM_OPTIM_UPDATES
wandb.config.optim_update_iter = OPTIM_ITER
wandb.config.tau = TAU
wandb.config.num_obs = NUM_OBS


# %%

# %%
Transition = namedtuple('Transition',

                        ('obs', 'action', 'next_obs', 'reward'))

from pathlib import Path

def save(agent, name):
        agent_file = Path(f"agent_{name}.m")
        torch.save(a.policy_net.state_dict(), agent_file)
# %%
def show_state(env, step=0):

    plt.figure(3, figsize=(10, 10))
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    # pause for plots to update
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.pause(0.0001)

# %%
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def flush(self):
        self.memory = []

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*sample))
        return batch

    def __len__(self):
        return len(self.memory)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# %%
class DuelingDQN(nn.Module):

    def __init__(self, obs_dim, num_actions=5):
        super(DuelingDQN, self).__init__()
        self.input_dim = obs_dim
        self.output_dim = num_actions
        self.hidden = HIDDEN_DIM

        self.lin_sin = nn.Linear(self.input_dim[0], self.hidden)

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], self.hidden),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden *2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.output_dim)
        )

    def forward(self, state):
        state = state.view(-1, self.input_dim[0])
        features = self.feauture_layer(state)
        sin_features = self.lin_sin(state)
        sin_features = torch.sin(sin_features)

        cat_features = torch.cat([features, sin_features], dim=1)
        values = self.value_stream(cat_features)
        advantages = self.advantage_stream(cat_features)
        qvals = values + (advantages - advantages.mean())

        return qvals.squeeze(1)

# %%
class QNet(nn.Module):

    def __init__(self, obs_dim=3,  num_actions=5):
        super().__init__()
        self.obs_dim = obs_dim
        hidden_dim = 100
        hidden_dim_2 = 50
        self.lin_1 = nn.Linear(obs_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs):
        x = obs.view(-1, self.obs_dim)
        x = self.lin_1(x)
        x = F.relu(x)
        return self.head(x)


# %%
class Agent(nn.Module):

    def __init__(self,
                num_actions=5,
                epsilon_start=1,
                capacity=CAPACITY,
                gamma=GAMMA,
                device=DEV,
                lr=LR,
                mem_obs=NUM_OBS,
                duelling=DUELLING,
                tau=TAU):

        super().__init__()
        self.duelling = duelling
        self.num_actions = num_actions
        self.dev = device
        self.episode_iter = 0
        self.not_done = True



        self.obs_dim = NUM_OBS * 4
        self.train_iter = 0
        self.mem_obs = mem_obs
        self.explore = True
        self.gamma = gamma
        self.observations = []
        self.loss = None
        self.loss_func = torch.nn.MSELoss()
        self.train = False

        self.tau = tau

        if self.duelling:
            self.policy_net = DuelingDQN(obs_dim=[self.obs_dim])
            self.target_net = DuelingDQN(obs_dim=[self.obs_dim])
        else:
            self.policy_net = QNet(obs_dim=self.obs_dim)
            self.target_net = QNet(obs_dim=self.obs_dim)

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.policy_net.to(device)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.episodes = 0
        self.memory = ReplayMemory(capacity)
        self.base_eps = 0

    def push(self, *args):

        if self.train:
            self.memory.push(*args)

    def done(self):
        self.episode_iter = 0
        self.episodes += 1
        self.observations = []

    @property
    def eps(self):
        eps  = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.episodes / EPS_DECAY)

        eps = find_nearest(EPS, eps)
        return eps

    def act(self, obs):
        obs = self.encode_obs(obs, store=True)

        sample = np.random.rand()

        if sample < self.eps and self.explore:
            action = np.random.randint(0, self.num_actions)
        elif sample < self.base_eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action = self.greedy(obs)

        self.episode_iter +=1

        if self.train:
            self.train_iter +=1

        return action

    def greedy(self, obs):
        with torch.no_grad():
            logits = self.policy_net(obs)
            action = logits.argmax().item()
            return action


    def optimizer_step(self, batch_size=BATCH_SIZE):

        if not self.train:
            return None

        batch = self.memory.sample(batch_size)

        obs = torch.cat(batch.obs)
        action = torch.cat(batch.action)
        next_obs = torch.cat(batch.next_obs)
        reward = torch.cat(batch.reward).squeeze()


        logits = self.policy_net(obs)
        action_values = logits.gather(1, action)
        next_logits = self.target_net(next_obs)
        max_action_value = next_logits.max(1)[0].detach()
        targets = max_action_value * self.gamma + reward
        loss = self.loss_func(action_values, targets.unsqueeze(1))

        self.loss = loss.item()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def encode_action(self, action):
        action = torch.tensor(action, device=self.dev).unsqueeze(0).unsqueeze(0)
        return action

    def encode_past_obs(self, observations):
        if len(observations) < self.mem_obs:
            obs_past = [observations[0]] * (self.mem_obs - len(observations)) + observations
        else:
            obs_past = observations[-self.mem_obs:]
        obs_past = [torch.tensor(obs, device=self.dev).unsqueeze(0) for obs in obs_past]
        obs_past = torch.cat(obs_past)
        return obs_past.unsqueeze(0)

    # def encode_obs(self, obs):
    #     obs = obs + [self.episode_iter]
    #     obs = torch.tensor(obs, device=self.dev).unsqueeze(0)
    #     return obs

    def encode_obs(self, obs, next_obs=False, store=True):
        obs = obs + [self.episode_iter /50,  self.train_iter/1000]

        observations = self.observations + [obs]

        if next_obs:
            obs_next = next_obs + [(self.episode_iter + 1) /50, (self.train_iter + 1)/1000]
            observations = observations + [obs_next]
        if store: self.observations = observations
        return self.encode_past_obs(observations)

    def encode_reward(self, reward):
        reward = torch.tensor(reward, device=self.dev).unsqueeze(0).unsqueeze(0)
        return reward

    def promote_target_net(self):

      for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):
            new_target_param = self.tau * param + ((1 - self.tau) * target_param)
            target_param.data.copy_(new_target_param)

        # self.target_net.load_state_dict(self.policy_net.state_dict())


def act(agents, obs_n):
    actions = [agents[i].act(obs_n[i]) for i in range(len(agents))]
    return actions

def store_in_mem(agents, actions, obs_n, next_obs_n, reward_n, done_counter):

    for i, agent in enumerate(agents):
        if done_counter[i] < 1:
            obs = agent.encode_obs(obs_n[i], store=False)
            next_obs = agent.encode_obs(obs_n[i], next_obs_n[i], store=False)
            action = agent.encode_action(actions[i])
            reward = agent.encode_reward(reward_n[i])
            agent.memory.push(obs, action, next_obs, reward)


def run_episode(agents, env, train=True, explore=True , active_agent=None,  show=False):

        done_n = [False for _ in range(env.n_agents)]
        rewards = []
        all_actions = []
        obs_n = env.reset()

        take_actions = partial(act, agents)

        store = partial(store_in_mem, agents)

        for a in agents:
            if a.train:
                a.explore = explore

        def make_way(agent):
            for i in [1, 1, 0, 0, 1, 1, 1, 0 ,0]:
                if agent == 0:
                    yield i
                else:
                    yield i * 3
            while True:
                yield 0

        # if PASSIVE_AGENT is not None:
        #     flee = make_way(PASSIVE_AGENT)
        #     next(flee)

        i = 0
        done_counter = np.zeros(len(agents))

        while not all(done_n):
            actions = take_actions(obs_n)

            # if PASSIVE_AGENT is not None:
            #     actions[PASSIVE_AGENT] = next(flee)

            all_actions.append(actions)
            next_obs_n, reward_n, done_n, info = env.step(actions)


            if train:
                store(actions, obs_n, next_obs_n, reward_n, done_counter)

            for a in agents:
                if TARGET_UPDATE_ITER and a.train_iter % TARGET_UPDATE_ITER == 0:
                    a.promote_target_net()


            if train and OPTIM_ITER:
                for i, a in enumerate(agents):
                    if a.train_iter % OPTIM_ITER == 0 and a.train_iter > BATCH_SIZE*2:
                        for j in range(NUM_OPTIM_UPDATES):
                            a.optimizer_step()
            if show:
                show_state(env)

            done_counter += np.array(done_n).astype(float)
            rewards.append(sum(reward_n))
            obs_n = next_obs_n

        for a in agents:
            a.done()

        return agents, sum(rewards), all_actions



# %%
def run(agents, env, epochs=EPISODES, thresh=100):

    train_rewards = [-10]
    test_rewards = [-10]
    test_avg_rwrds = [-10]

    test_reward = None
    progress = trange(epochs)

    for a in agents:
        a.train = True


    for i in progress:

        agents, eps_reward, actions = run_episode(agents, env,train=True, explore=True, active_agent=None, show=False)

        train_rewards.append(eps_reward)

        if i % 100:
            test_reward = test(agents, env, show=False)
            train_rewards.append(test_reward)
            test_avg_rwrds.append(np.mean(test_rewards[-100:]))
            wandb.log({"episode" : i, "test/reward" : test_reward , "test/rewards/max": max(train_rewards), "test/rewads/max_avg" :max(test_avg_rwrds)})


        loss_1, loss_2 = agents[0].loss, agents[1].loss
        eps = agents[0].eps

        wandb.log({"episode" : i,  "train/reward": eps_reward,  "train/reward/max" : max(train_rewards) , "loss_1" : loss_1, "loss_2": loss_2 , "eps_1": agents[0].eps, "eps_2": agents[1].eps})
        progress.set_postfix({ 'm_t': np.mean(train_rewards[-100:]), "m": np.mean(train_rewards[-100:]), "b": max(train_rewards), "b_t":max(train_rewards), "eps":eps, "l_1":loss_1,"l_2": loss_2 })

    for i, a in enumerate(agents):
        save(agent)

    return agents, train_rewards, train_rewards
# %%

def test(agents, env, active_agent=None, show=False):
    for a in agents:
        a.train = False
    agents, eps_reward, obs = run_episode(agents, env, train=False, explore=False, show=show)
    for a in agents:
        a.train = True
    env.close()
    return eps_reward


def plot_rewards(rewards, window=100):
    numbers_series = pd.Series(rewards)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    plt.plot(moving_averages)

def moving_average(nums, window=100):
    numbers_series = pd.Series(nums)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    return list(moving_averages)


def train_one_by_one():

    active = [1]
    passive = [0]
    episodes = 10000
    folds = 1400
    agents = [Agent(), Agent()]

    for f in range(int(episodes/folds)):
        for i in active: agents[i].train = True
        agents, rewards, train_rewards = run(agents, folds)
        active, passive = passive, active

    return agents


# %%

env = gym.make("Switch4-v0") # Use "Switch4-v0" for the Switch-4 game
agents = [Agent() for a in range(4)]
for i, a in enumerate(agents):
    save(a, i)

agents, train_rewards, train_rewards = run(agents, env)

for i, a in enumerate(agents):
    save(a, i)

# %% [markdown]
"""

"""

# # %%
# env = gym.make("Switch2-v0")
# env.reset()
# show_state(env)
# run_episode(agents, env, train=False, explore=False, show=True)

# agent = Agent()

# agent

P
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
import pickle
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

# %%
# PASSIVE_AGENT = None
DEV = "cuda:0"

# %%
# num_obs = 5
# duelling = true
# eps_start = 0.5
# eps_end = 0.01
# eps_decay = 1000
# eps = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
# gamma = 0.9
# lr = 0.001
# capacity = 1000
# optim_iter = 5
# target_update_iter = 5
# batch_size = 124
# hidden_dim = 25
# num_optim_updates = 2
# tau = 0.01
# episodes = 7000
# post_reward = 0
# env_steps = 50


# %%

# %%
Transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))

from pathlib import Path


def save(agents):
    agents_path = Path("agents4.pkl")
    a_file = open(agents_path, "wb")
    pickle.dump(agents, a_file)
    a_file.close()


def load_agents(agents_path: Path):
    a_file = open(agents_path, "rb")
    agents = pickle.load(a_file)
    return agents


# %%
def show_state(env, step=0):

    # plt.figure(3, figsize=(10, 10))
    # plt.clf()
    plt.imshow(env.render(mode="rgb_array"))
    # pause for plots to update
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.pause(0.0001)


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
    def __init__(self, obs_dim, hidden_dim, num_actions=5):
        super(DuelingDQN, self).__init__()
        self.input_dim = obs_dim
        self.output_dim = num_actions
        self.hidden = hidden_dim

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], self.hidden),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden, int(self.hidden / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden / 2), 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden, int(self.hidden / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden / 2), self.output_dim),
        )

    def forward(self, state):
        state = state.view(-1, self.input_dim[0])
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals.squeeze(1)


# %%
class QNet(nn.Module):
    def __init__(self, obs_dim=3, hidden_dim=50, num_actions=5):
        super().__init__()
        self.obs_dim = obs_dim
        hidden_dim = hidden_dim
        hidden_dim_2 = hidden_dim
        self.lin_1 = nn.Linear(obs_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.head = nn.Linear(int(hidden_dim / 2), num_actions)

    def forward(self, obs):
        x = obs.view(-1, self.obs_dim)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        x = F.relu(x)
        return self.head(x)

class ValueDecomp:
    def __init__(self, agents, capacity, gamma, lr):

        self.gamma = gamma
        self.memory = ReplayMemory(capacity)
        self.loss_func = nn.MSELoss()
        params = []
        self.agents = agents

        for a in agents:
            params += list(a.policy_net.parameters())

        self.optim = torch.optim.Adam(params, lr)

    def optimizer_step(self, batch_size):

        mems = [a.memory.memory for a in self.agents]
        combo = list(zip(*mems))
        sample = random.sample(combo, batch_size)
        batches = [Transition(*zip(*s)) for s in zip(*sample)]

        q_vals = []
        targets = []

        for i, agent in enumerate(self.agents):

            batch = batches[i]

            obs = torch.cat(batch.obs)
            action = torch.cat(batch.action)
            next_obs = torch.cat(batch.next_obs)
            reward = torch.cat(batch.reward).squeeze()

            logits = self.agents[i].policy_net(obs)
            action_value = logits.gather(1, action)
            next_logits = self.agents[i].target_net(next_obs)
            max_action_value = next_logits.max(1)[0].detach()
            target = max_action_value * self.gamma + reward

            q_vals.append(action_value)
            targets.append(target)

        q_val_global = torch.stack(q_vals).sum(0)
        target_global = torch.stack(targets).sum(0)

        loss = self.loss_func(q_val_global, target_global.unsqueeze(1))

        self.loss = loss.item()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss



class Agent(nn.Module):
    def __init__(
        self,
        num_obs,
        eps_start,
        eps_end,
        eps_decay,
        eps_list=[],
        tau=0.01,
        hidden_dim=25,
        duelling=True,
        device=DEV,
        capacity=1000,
        num_actions=5,
    ):

        super().__init__()
        self.duelling = duelling
        self.num_actions = num_actions
        self.dev = device

        self.episode_iter = 0
        self.episodes = 0
        self.train_iter = 0

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_list = eps_list

        self.obs_dim = num_obs * 3
        self.num_obs = num_obs

        self.observations = []
        self.loss = None

        self.tau = tau

        if self.duelling:
            self.policy_net = DuelingDQN(obs_dim=[self.obs_dim], hidden_dim=hidden_dim)
            self.target_net = DuelingDQN(obs_dim=[self.obs_dim], hidden_dim=hidden_dim)
        else:
            self.policy_net = QNet(obs_dim=self.obs_dim, hidden_dim=hidden_dim)
            self.target_net = QNet(obs_dim=self.obs_dim, hidden_dim=hidden_dim)

        self.policy_net.to(device)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(capacity)

    def finish_episode(self):
        self.episode_iter = 0
        self.episodes += 1
        self.observations = []

    def epsilon(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.train_iter / self.eps_decay
        )

        if self.eps_list:
            eps = find_nearest(self.eps_list, eps)

        return eps

    def act(self, obs, explore=True):
        obs = self.encode_obs(obs, store=True)
        sample = np.random.rand()

        if sample < self.epsilon() and explore:
            action = np.random.randint(0, self.num_actions)
        else:
            action = self.greedy(obs)

        self.episode_iter += 1
        if explore:
            self.train_iter += 1

        return action

    def greedy(self, obs):
        with torch.no_grad():
            logits = self.policy_net(obs)
            action = logits.argmax().item()
            return action

    def encode_action(self, action):
        action = torch.tensor(action, device=self.dev).unsqueeze(0).unsqueeze(0)
        return action

    def encode_obs_frame(self, observations):
        if len(observations) < self.num_obs:
            obs_past = [observations[0]] * (
                self.num_obs - len(observations)
            ) + observations
        else:
            obs_past = observations[-self.num_obs :]
        obs_past = [torch.tensor(obs, device=self.dev).unsqueeze(0) for obs in obs_past]
        obs_past = torch.cat(obs_past)
        return obs_past.unsqueeze(0)

    def encode_obs(self, obs, next_obs=False, store=True):
        obs = obs + [
            self.episode_iter / 50,
        ]

        observations = self.observations + [obs]

        if next_obs:
            obs_next = next_obs + [(self.episode_iter + 1) / 50]
            observations = observations + [obs_next]

        if store:
            self.observations = observations

        return self.encode_obs_frame(observations)

    def encode_reward(self, reward):
        reward = torch.tensor(reward, device=self.dev).unsqueeze(0).unsqueeze(0)
        return reward

    def promote_target_net(self):

        for target_param, param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            new_target_param = self.tau * param + ((1 - self.tau) * target_param)
            target_param.data.copy_(new_target_param)


def act(agents, obs_n, explore=True):
    actions = [agents[i].act(obs_n[i], explore) for i in range(len(agents))]
    return actions


def push_to_replay_buffers(
    agents, post_reward, actions, obs_n, next_obs_n, reward_n, done_counter
):

    for i, agent in enumerate(agents):
        if done_counter[i] > 1:
            reward_n[i] = post_reward
        obs = agent.encode_obs(obs_n[i], store=False)
        next_obs = agent.encode_obs(obs_n[i], next_obs_n[i], store=False)
        action = agent.encode_action(actions[i])
        reward = agent.encode_reward(reward_n[i])
        agent.memory.push(obs, action, next_obs, reward)


def episode(env, take_actions):
    total_reward = 0
    done_counter = np.zeros(env.n_agents)
    home_n = 0
    done_n = [False for _ in range(env.n_agents)]
    obs_n = env.reset()
    while not all(done_n):
        actions = take_actions(obs_n)
        next_obs_n, reward_n, done_n, info = env.step(actions)
        done_counter += np.array(done_n).astype(float)
        home_n += 5 in reward_n
        total_reward += sum(reward_n)
        yield obs_n, next_obs_n, actions, reward_n, done_counter, home_n, total_reward
        obs_n = next_obs_n


def train(
    env="Switch4-v0",
    num_episodes=2000,
    num_obs=5,
    num_agents=2,
    duelling=True,
    eps_start=0.5,
    eps_end=0.01,
    eps_decay=1000,
    eps_list=[],
    gamma=0.9,
    capacity=1000,
    optim_iter=5,
    num_optim_updates=2,
    target_update_iter=5,
    post_reward=0,
    batch_size=124,
    lr=0.001,
    hidden_dim=25,
    tau=0.01,
    episodes=7000,
    env_steps=50,
):

    agents = [
        Agent(
            num_obs, eps_start, eps_end, eps_decay, eps_list, tau, hidden_dim, duelling
        )
        for _ in range(num_agents)
    ]


    test_rewards = []
    train_rewards = []

    train_iter = 0
    env = gym.make(env)

    act_train = partial(act, agents, explore=True)
    push = partial(push_to_replay_buffers, agents, post_reward)

    optimiser = ValueDecomp(agents, capacity, gamma, lr)

    for episode_num in trange(num_episodes):

        # Run training episode
        for obs_n, next_obs_n, actions, reward_n, done_counter, home_n, total_reward in episode(env, act_train):

            push(actions, obs_n, next_obs_n, reward_n, done_counter)

            if not train_iter % target_update_iter and target_update_iter:
                for a in agents:
                    a.promote_target_net()

            if not train_iter % optim_iter and train_iter > batch_size:
                for update in range(num_optim_updates):
                    loss = optimiser.optimizer_step(batch_size)
                    wandb.log({"episode": episode_num, "loss": loss})

            train_iter += 1

        for a in agents: a.finish_episode()

        # log training episode results
        train_rewards.append(total_reward)
        wandb.log(
            {
                "episode": episode_num,
                "train/reward": total_reward,
                "train/num_home": home_n,
                "train/reward/max": max(train_rewards),
                "eps": agents[0].epsilon(),
            }
        )

        # evaluate
        if not episode_num % 50:
            test_reward, home_n_test = test(agents, env)
            test_rewards.append(test_reward)

            wandb.log(
                {
                    "episode": episode_num,
                    "test/num_home": home_n_test,
                    "test/reward": test_reward,
                    "test/rewards/max": max(test_rewards),
                }
            )

    return agents, train_rewards, test_rewards



def test(agents, env):
    act_test = partial(act, agents, explore=False)
    *all_but_last, last_result = episode(env, act_test)
    *_, home_n_test, test_reward = last_result
    for a in agents: a.finish_episode()
    return test_reward, home_n_test


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


# %%
if __name__ == "__main__":

    config = {
        "env": "Switch4-v0",
        "num_episodes": 4000,
        "num_obs": 1,
        "num_agents": 4,
        "duelling": True,
        "eps_start": 0.5,
        "eps_end": 0.01,
        "eps_decay": 40000,  # now on train iter not episode
        "eps_list": [],
        "gamma": 0.9,
        "capacity": 4000,
        "optim_iter": 1,
        "num_optim_updates": 5,
        "target_update_iter": 1,
        "batch_size": 64,
        "lr": 0.001,
        "hidden_dim": 20,
        "tau": 0.005,
        "env_steps": 50,
        "post_reward": 5,
    }

    cfg = {"notes": "half hidden dim in downstream layers"}
    cfg.update(config)
    wandb.init(project="switch4_scrap", config=cfg)

    train(**config)

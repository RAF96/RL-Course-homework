import copy
from collections import deque

import tqdm as tqdm
from gym import make
import numpy as np
import torch
import torch.nn.functional as F
import random

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from hw03_pendulum.validate import validate

N_STEP = 1
GAMMA = 0.99


def transform_state(state):
    return np.array(state)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim
        l1_dim = 50
        self._l1 = nn.Linear(state_dim, l1_dim)
        l2_dim = 50
        self._l2 = nn.Linear(l1_dim, l2_dim)
        self._l3 = nn.Linear(l2_dim, 2)

    def forward(self, x):
        x = self._l1(x)
        x = F.relu(x)
        x = self._l2(x)
        x = F.relu(x)
        x = self._l3(x)

        def function(x):
            mu = x[:, 0]
            sigma = x[:, 1]
            sigma = F.softplus(sigma)
            return torch.stack((mu, sigma), dim=1)
        x = function(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim
        l1_dim = 50
        self._l1 = nn.Linear(state_dim + action_dim, l1_dim)
        l2_dim = 50
        self._l2 = nn.Linear(l1_dim, l2_dim)
        self._l3 = nn.Linear(l2_dim, 1)

    def forward(self, x):
        x = self._l1(x)
        x = F.relu(x)
        x = self._l2(x)
        x = F.relu(x)
        x = self._l3(x)
        return x


class A2C:
    def __init__(self, state_dim, action_dim, tensorboard_writer):
        self._gamma = GAMMA ** N_STEP

        self._actor = Actor(state_dim, action_dim)
        self._critic = Critic(state_dim, action_dim)

        self._critic_loss = nn.MSELoss(reduction="none")  # MOCK, may be should be changed

        self._lr = 0.001
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=self._lr * 0.1, weight_decay=0.1)
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=self._lr, weight_decay=0.1)

        self._learning_step = 0
        self._tensorboard_writer = tensorboard_writer

    def update(self, transition):
        self._learning_step += 1

        state, action, next_state, reward, done = transition

        state = torch.tensor(state, dtype=torch.float).view((1, -1))
        action = torch.tensor(action, dtype=torch.float).view((1, -1))
        next_state = torch.tensor(next_state, dtype=torch.float).view((1, -1))
        reward = torch.tensor(reward).view((1, -1))
        done = torch.tensor(done, dtype=torch.int).view((1, -1))

        with torch.no_grad():
            next_action_mu_and_sigma = self._actor(next_state)
            next_action_mu = next_action_mu_and_sigma[:, 0]
            next_action_sigma = next_action_mu_and_sigma[:, 1]
            next_action = torch.tensor(np.random.normal(next_action_mu.item(), next_action_sigma.item()), dtype=torch.float).view((1, -1))

        Q_eval = self._critic(torch.cat((state, action), 1))
        with torch.no_grad():
            next_state_and_action = torch.cat((next_state, next_action), 1)
            Q_target = reward + self._gamma * self._critic(next_state_and_action) * (1 - done)
        critic_loss = self._critic_loss(Q_eval, Q_target)

        # обновить critic
        mu_and_sigma = self._actor(state)
        mu = mu_and_sigma[:, 0]
        sigma = mu_and_sigma[:, 1]

        # actor_loss = torch.log(self._actor(state)) * Q_eval.item()
        actor_loss = (action - mu) / torch.clamp(sigma, 0.001)

        assert not torch.isnan(actor_loss).any()
        assert not torch.isnan(critic_loss).any()

        self._tensorboard_writer.add_scalar('hw3/actor_loss', actor_loss.item(), self._learning_step)
        self._tensorboard_writer.add_scalar('hw3/critic_loss', critic_loss.item(), self._learning_step)
        self._tensorboard_writer.add_scalar('hw3/actor_mu', mu.item(), self._learning_step)
        self._tensorboard_writer.add_scalar('hw3/actor_sigma^2', sigma.item(), self._learning_step)

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).view((1, -1))
            mu_and_sigma = self._actor(state)
            mu = mu_and_sigma[:, 0]
            sigma = mu_and_sigma[:, 1]
            action = np.random.normal(mu, sigma)
            assert action != np.nan
            return action

    def save(self, version):
        torch.save(self._actor, f"models/agent{version}.pkl")


def train(writer):
    env = make("Pendulum-v0")
    algo = A2C(state_dim=3, action_dim=1, tensorboard_writer=writer)
    episodes = 10000

    for i in tqdm.trange(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            action = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                algo.update((state_buffer[0], action_buffer[0], next_state,
                             sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                algo.update((state_buffer[k], action_buffer[k], next_state,
                             sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if i % 10 == 0:
            algo.save(i)
            writer.add_scalar("hw3/mean_total_reward", validate(algo, env), i)


if __name__ == "__main__":
    with SummaryWriter(log_dir='runs', purge_step=0) as writer:
        train(writer)

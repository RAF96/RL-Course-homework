from gym import make
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
from collections import deque
import random
from tqdm import trange

N_STEP = 1
GAMMA = 0.9

TRANSFORM_TO_FEATURE = True
STATE_SPACE_SIZE = None
RADIAN_POINTS = np.linspace(-1, 1, 5)


def normalize_state(state):
    # [-1.2, 0.6] [-0.07, 0.07]
    state[0] = ((state[0] + 1.2) / (1.2 + 0.6) - 0.5) * 2
    state[1] = (state[1]) / 0.07
    return state


def transform_state(state):
    # result = []
    # result.extend(state)
    # return np.array(result)
    global STATE_SPACE_SIZE

    state = normalize_state(state)

    if not TRANSFORM_TO_FEATURE:
        STATE_SPACE_SIZE = 2
        return state
    else:

        result = []
        clone = state.clone()
        clone[1] = clone[1] * 20
        result.extend(clone)
        result.extend(torch.abs(clone))
        # for x in RADIAN_POINTS:
        #     for speed in RADIAN_POINTS:
        #         result.append(np.exp(-(state[0] - x) ** 2 - (state[1] - speed) ** 2))

        STATE_SPACE_SIZE = len(result)
        torch_result = torch.tensor(result)
        return torch_result


MODIFIED_REWARD = True


def modified_reward(state, new_state, reward):
    if MODIFIED_REWARD:
        return reward + 300 * (GAMMA * abs(new_state[1]) - abs(state[1])) / 0.14 / 2
    else:
        return reward


def to2d(state):
    if len(state.shape) == 0:
        return state.view((1, 1))
    elif len(state.shape) == 1:
        return state.view((1, -1))
    elif len(state.shape) == 2:
        return state
    else:
        raise RuntimeError(f"Case of the state shape: {state.shape} undefined")


class AQL:
    def __init__(self, state_dim, action_dim, tensor_board_writer, lr=0.001):
        self._learn_step_counter = 0
        self._tensor_board_writer = tensor_board_writer
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._gamma = GAMMA ** N_STEP
        self._lr = lr
        self._weight = torch.zeros((state_dim, action_dim), dtype=torch.double, requires_grad=True)
        self._bias = torch.zeros((1, action_dim), dtype=torch.double, requires_grad=True)
        self._loss = nn.MSELoss()
        self._optim = torch.optim.Adam([self._weight, self._bias], lr=lr, weight_decay=0.1)

    def update(self, transition):
        self._learn_step_counter += 1
        state, action, next_state, reward, done = transition
        state, next_state, reward = map(lambda x: torch.tensor(x, dtype=torch.double), (state, next_state, reward))
        state, next_state = map(transform_state, [state, next_state])
        state, next_state, reward = map(to2d, (state, next_state, reward))
        Q_function = next_state.matmul(self._weight) + self._bias
        target = reward + self._gamma * torch.max(Q_function).view((1, 1))
        loss = self._loss(Q_function[0, action], target.view(()))

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        self._tensor_board_writer.add_scalar("loss", loss.detach().item(), self._learn_step_counter)

    def act(self, state, target=False):
        state = torch.tensor(state)
        state = transform_state(state)
        state = to2d(state)
        with torch.no_grad():
            Q_function = torch.matmul(state, self._weight) + self._bias
            return np.argmax(Q_function.squeeze()).item()

    def save(self):
        weight = self._weight.detach().numpy()
        bias = self._bias.detach().numpy()
        np.savez("agent.npz", weight, bias)


def create_generator_eps(start=0.1, finish=0.05, num_of_iter=8000):
    delta = (start - finish) / num_of_iter
    current = start
    while True:
        yield max(current, finish)
        current -= delta


if __name__ == "__main__":
    with SummaryWriter(log_dir="runs/hw1", purge_step=0) as writer:
        if STATE_SPACE_SIZE is None:
            transform_state(torch.tensor([0, 0]))  # Mock
        env = make("MountainCar-v0")
        aql = AQL(state_dim=STATE_SPACE_SIZE, action_dim=3, tensor_board_writer=writer)
        generator_eps = create_generator_eps()
        episodes = 200

        for i in trange(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            reward_buffer = deque(maxlen=N_STEP)
            state_buffer = deque(maxlen=N_STEP)
            action_buffer = deque(maxlen=N_STEP)
            while not done:
                if random.random() < next(generator_eps):
                    action = env.action_space.sample()
                else:
                    action = aql.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = modified_reward(state, next_state, reward)
                total_reward += reward
                steps += 1
                reward_buffer.append(reward)
                state_buffer.append(state)
                action_buffer.append(action)
                if len(reward_buffer) == N_STEP:
                    aql.update((state_buffer[0], action_buffer[0], next_state,
                                sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
                state = next_state
            if len(reward_buffer) < N_STEP:
                rb = list(reward_buffer)
                for k in range(1, N_STEP):
                    aql.update((state_buffer[k], action_buffer[k], next_state,
                                sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

            writer.add_scalar("reward", total_reward, i)

            if i % 20 == 0:
                aql.save()

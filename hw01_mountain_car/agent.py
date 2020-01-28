import random
import numpy as np
import os

import torch

from .train import transform_state, to2d


class Agent:
    def __init__(self):
        npzfile = np.load(__file__[:-8] + "/agent.npz")
        self._weight, self._bias = npzfile['arr_0'], npzfile['arr_1']
        self._weight = to2d(torch.as_tensor(self._weight))
        self._bias = to2d(torch.as_tensor(self._bias))
        print(self._weight)
        print(self._bias)
        # self._weight = to2d(torch.as_tensor(np.array([[0, 0, 0], [-1, 0, 1]], dtype=np.float)))
        # self._bias = to2d(torch.as_tensor(np.array([0, 0, 0], dtype=np.float)))

    def act(self, state):
        state = torch.tensor(state)
        state = transform_state(state)
        state = to2d(state)
        with torch.no_grad():
            Q_function = torch.matmul(state, self._weight) + self._bias
            print(state)
            print(Q_function)
            return np.argmax(Q_function.squeeze()).item()
        # if state[1] < 0:
        #     return 0
        # else:
        #     return 2

    def reset(self):
        pass


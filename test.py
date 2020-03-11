
import gym
import torch

from hw03_pendulum.agent import Agent
from hw03_pendulum.validate import validate


class TestAgent:
    def __init__(self, env: gym.Env):
        self._env = env

    def act(self, state: torch.Tensor):
        return self._env.action_space.sample()


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    # agent = Agent()
    agent = TestAgent(env)
    print(f"Mean reward: {round(validate(agent, env, 1, True), 2)}")
    env.close()

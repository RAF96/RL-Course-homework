import time

import gym

from hw01_mountain_car.agent import Agent

if __name__ == "__main__":
    aql = Agent()
    env = gym.make("MountainCar-v0")

    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        env.render()
        time.sleep(0.01)
        action = aql.act(state)
        print(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(total_reward)
    env.close()

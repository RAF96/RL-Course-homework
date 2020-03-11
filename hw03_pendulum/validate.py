from time import sleep


def validate_one_case(agent, env, verbose=False):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if verbose:
            print(f"Action {action}, reward: {round(reward, 2)}")
            env.render()
            sleep(0.01)
    return total_reward


def validate(agent, env, num_of_episode=75, verbose=False):
    rewards = []
    for _ in range(num_of_episode):
        reward = validate_one_case(agent, env, verbose)
        rewards.append(reward)
    return sum(rewards) / num_of_episode

import numpy as np
from environment import AsteroidEnv
from dqn_agent import DQNAgent
import time

env = AsteroidEnv(render_mode=False)
agent = DQNAgent(input_dim=10, output_dim=5)

episodes = 500
target_update_freq = 10

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.memorize(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward

        # For visualization, turn on:
        # env.render()

    if ep % target_update_freq == 0:
        agent.update_target()

    print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

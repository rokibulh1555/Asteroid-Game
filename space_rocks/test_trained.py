from environment import AsteroidEnv
from dqn_agent import DQNAgent
import torch

agent = DQNAgent(input_dim=10, output_dim=5)
agent.model.load_state_dict(torch.load("dqn_model.pth"))
agent.model.eval()

env = AsteroidEnv(render_mode=True)

for _ in range(10):  # Run 10 episodes
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        env.render()

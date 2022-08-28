import copy
import collections
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('..')
import matplotlib.pyplot as plt

from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper

def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent(test=True)
    agent.train()
    agent.loadModel(load_model_pre)
    states, obs = envs.reset()
    test_episode = 100
    total = 0
    s = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < test_episode:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        if actions_star[0][0]<0.9:
            actions_star[0][0] = torch.tensor([0])
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        s += rewards.int().sum().item()

        if dones.sum():
            total += dones.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(dones.sum().int().item())

if __name__ == '__main__':
    test()
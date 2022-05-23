import matplotlib.pyplot as plt
import torch
import numpy as np

def visualizeBC(agent, sim_obs, actions):
    sim_obs0 = sim_obs[0]
    sim_obs1 = sim_obs[1]
    sim_obs2 = sim_obs[2]
    sim_obs_new = sim_obs[3]

    sim_actions1_star_idx = actions[0]
    sim_actions_new_star_idx = actions[1]

    fig, axes = plt.subplots(1,4)
    axes[0].imshow(sim_obs0[0])
    axes[0].set_title("sim_obs0")
    axes[1].imshow(sim_obs1[0])
    unscaled, sim_action = agent.decodeSingleActions(*[torch.tensor(sim_actions1_star_idx)[i] for i in range(5)])
    axes[1].arrow(x=64, y=64, dx=sim_action[1]/0.3*128, dy=sim_action[2]/0.3*128, width=.8) 
    axes[1].set_title("sim_obs1")
    

    axes[2].imshow(sim_obs_new[0][0])
    unscaled, sim_action = agent.decodeSingleActions(*[torch.tensor(sim_actions_new_star_idx)[0,i] for i in range(5)])
    axes[2].arrow(x=64, y=64, dx=sim_action[1]/0.3*128, dy=sim_action[2]/0.3*128, width=.8) 
    axes[2].set_title("sim_obs_new")
    

    axes[3].imshow(sim_obs2[0])
    axes[3].set_title("sim_obs2")
    return fig
    


def visualizeExpert(agent, local_transition):

    column_num = 4
    fig, axes = plt.subplots(np.ceil(len(local_transition)/column_num).astype(int),column_num)
    for i in range(len(local_transition)):
        obs = local_transition[i].obs
        state = local_transition[i].state
        action = local_transition[i].action

        current_ax=axes[i//column_num, i%column_num]  
        current_ax.imshow(obs[0])
        unscaled, sim_action = agent.decodeSingleActions(*[torch.tensor(action)[i] for i in range(5)])
        print(f"action{i}", sim_action)
        current_ax.arrow(x=64, y=64, dx=sim_action[2]/0.3*128, dy=sim_action[1]/0.3*128, width=.8)
        current_ax.text(0, 0, u"\u2191", rotation = sim_action[4]*180/np.pi)
        if sim_action.dtype is torch.float64:
            current_ax.set_title("sim_obs")
    plt.show()

    return fig

if __name__ == "__main__":
    # # arrow test
    # pixels = np.ones([128, 128])
    # plt.imshow(pixels)
    # plt.arrow(x=64, y=64, dx=-10, dy=10, width=2)
    # plt.show()

    # text test
    pixels = np.ones([128, 128])
    plt.imshow(pixels)
    plt.text(0, 0, u"\u2191", rotation=-0.3*180/np.pi)
    plt.show()
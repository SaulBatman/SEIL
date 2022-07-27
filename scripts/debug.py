import matplotlib.pyplot as plt
import torch
import numpy as np

def visualizeTransitionTS(sim_obs, actions):

    sim_obs1 = sim_obs[1]
    sim_obs2 = sim_obs[2]
    sim_obs_new = sim_obs[3]

    sim_actions1_star = actions[0]
    sim_actions_new_star = actions[1]

    fig, axes = plt.subplots(1,3)
    axes[0].imshow(sim_obs1[0])
    axes[0].arrow(x=64, y=64, dx=sim_actions1_star[2]/0.45*128, dy=sim_actions1_star[1]/0.45*128, width=.5) 
    axes[0].set_title("obs1")
    axes[0].text(0, 0, f"action: {np.round(sim_actions1_star.tolist(), 2)}")
    

    axes[1].imshow(sim_obs_new[0][0])
    axes[1].arrow(x=64, y=64, dx=sim_actions_new_star[2]/0.45*128, dy=sim_actions_new_star[1]/0.45*128, width=.5) 
    axes[1].set_title("sim_obs_new")
    axes[1].text(0, 0, f"action: {np.round(sim_actions_new_star.tolist(), 2)}")
    
    axes[2].imshow(sim_obs2[0])
    axes[2].set_title("obs2")
    print(1)
    
    return fig
    
def visualizeTransition(agent, obs, actions):
    obs0 = obs[0]
    obs1 = obs[1]

    sim_actions1_star_idx = actions[0]
    

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(obs0[0])
    axes[0].set_title("obs0")
    unscaled, sim_action = agent.decodeSingleActions(*[torch.tensor(sim_actions1_star_idx)[i] for i in range(5)])
    print("action: ", sim_action)
    axes[0].arrow(x=64, y=64, dx=sim_action[2]/0.45*128, dy=sim_action[1]/0.45*128, width=.8) 
    

    axes[1].imshow(obs1[0])
    axes[1].set_title("obs1")
    print(1)

    return fig


def visualizeTraj(agent, local_transition, column_num = 4, delete_num=0):
    SMALL_SIZE = 8

    
    fig, axes = plt.subplots(np.ceil(len(local_transition)/column_num).astype(int),column_num)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.subplots_adjust(top = 0.90, bottom=0.15, right=0.95, left=0.08, hspace=0.4, wspace=0.2)
    
    for i in range(len(local_transition)):
        obs = local_transition[i].obs
        state = local_transition[i].state
        action = local_transition[i].action
        if len(local_transition)>column_num:
            current_ax=axes[i//column_num, i%column_num]  
        else:
            current_ax=axes[i%column_num] 
        current_ax.imshow(obs[0])
        unscaled, sim_action = agent.decodeSingleActions(*[torch.tensor(action)[i] for i in range(5)])
        print(f"action{i}", sim_action)
        current_ax.arrow(x=64, y=64, dx=sim_action[2]/0.3*128, dy=sim_action[1]/0.3*128, width=.8, color='y')
        current_ax.text(0, 0, u"\u2191", rotation = sim_action[4]*180/np.pi)
        current_ax.axis('off')
        if sim_action.dtype is torch.float64:
            current_ax.set_title("sim_obs")
        else:
            current_ax.set_title("expert_obs_ "+str(i))
    if delete_num > 0:
        for i in range(delete_num):
            fig.delaxes(axes[len(local_transition)//column_num, column_num-1-i])
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
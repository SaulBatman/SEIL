from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class SACVec(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning)

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']

        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def getSACAction(self, state, obs, evaluate):
        with torch.no_grad():
            if evaluate is False:
                action, _, _ = self.actor.sample(obs.to(self.device))
            else:
                _, _, action = self.actor.sample(obs.to(self.device))
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

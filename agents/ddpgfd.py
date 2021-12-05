from agents.ddpg import DDPG
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class DDPGfD(DDPG):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 demon_w=0.1, demon_l='mean'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.demon_w = demon_w
        assert demon_l in ['mean', 'pi']
        self.demon_l = demon_l

    def calcActorLoss(self):
        policy_loss = super().calcActorLoss()
        pi = self.loss_calc_dict['pi']
        mean = self.loss_calc_dict['mean']
        action = self.loss_calc_dict['action_idx']
        is_experts = self.loss_calc_dict['is_experts']
        # add expert loss
        if is_experts.sum():
            if self.demon_l == 'pi':
                demon_loss = F.mse_loss(pi[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss
            else:
                demon_loss = F.mse_loss(mean[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss
        return policy_loss


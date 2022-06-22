from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from agents.dqn_agent_fac import DQNAgentFac
from utils.parameters import heightmap_size, crop_size
from utils.torch_utils import centerCrop

class BCFac(DQNAgentFac):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(states, obs)

        p_loss = F.cross_entropy(q_p, p_id)
        dxy_loss = F.cross_entropy(q_dxy, dxy_id)
        dz_loss = F.cross_entropy(q_dz, dz_id)
        dtheta_loss = F.cross_entropy(q_dtheta, dtheta_id)

        loss = p_loss + dxy_loss + dz_loss + dtheta_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_calc_dict = {}

        return loss.item(), torch.tensor(0.)

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_p_prime, q_dxy_prime, q_dz_prime, q_dtheta_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            if self.com == 'add':
                q_prime = q_p_prime.max(1)[0] + q_dxy_prime.max(1)[0] + q_dz_prime.max(1)[0] + q_dtheta_prime.max(1)[0]
            elif self.com == 'mul':
                q_prime = q_p_prime.max(1)[0] * q_dxy_prime.max(1)[0] * q_dz_prime.max(1)[0] * q_dtheta_prime.max(1)[0]
            else:
                raise NotImplementedError
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(states, obs)
        q_p_pred = q_p[torch.arange(batch_size), p_id]
        q_dxy_pred = q_dxy[torch.arange(batch_size), dxy_id]
        q_dz_pred = q_dz[torch.arange(batch_size), dz_id]
        q_dtheta_pred = q_dtheta[torch.arange(batch_size), dtheta_id]
        if self.com == 'add':
            q_pred = q_p_pred + q_dxy_pred + q_dz_pred + q_dtheta_pred
        elif self.com == 'mul':
            q_pred = q_p_pred * q_dxy_pred * q_dz_pred * q_dtheta_pred
        else:
            raise NotImplementedError
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        self.loss_calc_dict['q_output'] = q_p, q_dxy, q_dz, q_dtheta
        self.loss_calc_dict['q_pred'] = q_p_pred, q_dxy_pred, q_dz_pred, q_dtheta_pred
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class DQNAgentFac(BaseAgent):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q_p, q_dxy, q_dz, q_dtheta = net(stacked.to(self.device))
        if to_cpu:
            q_p = q_p.to('cpu')
            q_dxy = q_dxy.to('cpu')
            q_dz = q_dz.to('cpu')
            q_dtheta = q_dtheta.to('cpu')
        return q_p, q_dxy, q_dz, q_dtheta

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(state, obs, to_cpu=True)
            p_id = torch.argmax(q_p, 1)
            dxy_id = torch.argmax(q_dxy, 1)
            dz_id = torch.argmax(q_dz, 1)
            dtheta_id = torch.argmax(q_dtheta, 1)

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_p = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_p.size(1))
        p_id[rand_mask] = rand_p.long()
        rand_dxy = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dxy.size(1))
        dxy_id[rand_mask] = rand_dxy.long()
        rand_dz = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dz.size(1))
        dz_id[rand_mask] = rand_dz.long()
        rand_dtheta = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dtheta.size(1))
        dtheta_id[rand_mask] = rand_dtheta.long()
        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_p_prime, q_dxy_prime, q_dz_prime, q_dtheta_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_p_prime.max(1)[0] + q_dxy_prime.max(1)[0] + q_dz_prime.max(1)[0] + q_dtheta_prime.max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(states, obs)
        q_p_pred = q_p[torch.arange(batch_size), p_id]
        q_dxy_pred = q_dxy[torch.arange(batch_size), dxy_id]
        q_dz_pred = q_dz[torch.arange(batch_size), dz_id]
        q_dtheta_pred = q_dtheta[torch.arange(batch_size), dtheta_id]
        q_pred = q_p_pred + q_dxy_pred + q_dz_pred + q_dtheta_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error

from agents.dqn_agent_com import DQNAgentCom
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils import torch_utils

class FCNDQN(DQNAgentCom):
    def __init__(self, workspace, heightmap_size, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, tau=0.01):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p=2, n_theta=3)
        self.workspace = workspace
        self.heightmap_size = heightmap_size
        workspace_size = workspace[0][1] - workspace[0][0]
        self.heightmap_resolution = workspace_size / heightmap_size

        self.tau = tau

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        self.p_range = torch.tensor([0, 1])
        self.d_theta_range = torch.tensor([-dr, 0, dr])
        self.dx = dx
        self.dy = dy
        self.dz_range = torch.tensor([-dz, 0, dz])

    def decodeActions(self, p_id, pixels, dz_id, dtheta_id):
        dx = ((pixels[:, 0].float() - self.heightmap_size//2) * self.heightmap_resolution)
        dy = ((pixels[:, 1].float() - self.heightmap_size//2) * self.heightmap_resolution)
        dx = torch.clip(dx, -self.dx, self.dx)
        dy = torch.clip(dy, -self.dy, self.dy)
        p = self.p_range[p_id]
        dz = self.dz_range[dz_id]
        dtheta = self.d_theta_range[dtheta_id]
        actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
        action_idxes = torch.stack([p_id, pixels[:, 0], pixels[:, 1], dz_id, dtheta_id], dim=1)
        return action_idxes, actions

    def getActionFromPlan(self, plan):
        primitive = plan[:, 0:1]
        dx = plan[:, 1:2]
        dy = plan[:, 2:3]
        dz = plan[:, 3:4]
        dr = plan[:, 4:5]

        p_id = torch.argmin(torch.abs(self.p_range - primitive), 1)
        pixel_x = (dx / self.heightmap_resolution).long() + self.heightmap_size//2
        pixel_y = (dy / self.heightmap_resolution).long() + self.heightmap_size//2
        pixel_x = torch.clip(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clip(pixel_y, 0, self.heightmap_size-1)
        dz_id = torch.argmin(torch.abs(self.dz_range - dz), 1)
        dtheta_id = torch.argmin(torch.abs(self.d_theta_range - dr), 1)

        return self.decodeActions(p_id, torch.cat((pixel_x, pixel_y), dim=1), dz_id, dtheta_id)

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q = net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_p, self.n_z, self.n_theta, self.heightmap_size, self.heightmap_size)\
            .permute(0, 1, 4, 5, 2, 3)
        return q

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            q = self.forwardNetwork(state, obs, to_cpu=True)
            argmax = torch_utils.argmax5d(q)
            p_id = argmax[:, 0]
            dxy_id = argmax[:, 1:3]
            dz_id = argmax[:, 3]
            dtheta_id = argmax[:, 4]

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_p = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_p)
        p_id[rand_mask] = rand_p.long()
        # rand_dxy = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_xy)
        # dxy_id[rand_mask] = rand_dxy.long()
        rand_dz = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_z)
        dz_id[rand_mask] = rand_dz.long()
        rand_dtheta = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_theta)
        dtheta_id[rand_mask] = rand_dtheta.long()
        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1:3]
        dz_id = action_idx[:, 3]
        dtheta_id = action_idx[:, 4]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size, -1).max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q = self.forwardNetwork(states, obs)
        q_pred = q[torch.arange(batch_size), p_id, dxy_id[:, 0], dxy_id[:, 1], dz_id, dtheta_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error
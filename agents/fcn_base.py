from agents.dqn_agent_com import DQNAgentCom
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils import torch_utils

class FCNBase(DQNAgentCom):
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
        raise NotImplementedError

    def getEGreedyActions(self, state, obs, eps):
        raise NotImplementedError

    def calcTDLoss(self):
        raise NotImplementedError
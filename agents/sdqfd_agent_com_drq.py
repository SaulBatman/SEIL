from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from agents.dqn_agent_com_drq import DQNAgentComDrQ
from utils import torch_utils

class SDQfDComDrQ(DQNAgentComDrQ):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1,
                 l=0.1, w=0.1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.margin_l = l
        self.margin_weight = w

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        batch_size = batch_size * self.M
        is_experts = is_experts.repeat(self.M)
        q_output = self.loss_calc_dict['q_output']
        q_pred = self.loss_calc_dict['q_pred']

        if is_experts.sum() == 0:
            return torch.tensor(0)
        q_output = q_output.reshape(batch_size, -1)
        margin_losses = []
        for j in range(batch_size):
            if not is_experts[j]:
                margin_losses.append(torch.tensor(0).float().to(q_output.device))
                continue

            qe = q_pred[j]
            q_all = q_output[j]
            over = q_all[q_all > qe - self.margin_l]
            if over.shape[0] == 0:
                margin_losses.append(torch.tensor(0).float().to(q_output.device))
            else:
                over_target = torch.ones_like(over) * qe - self.margin_l
                margin_losses.append((over - over_target).mean())
        margin_loss = torch.stack(margin_losses).mean()
        return margin_loss

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        margin_loss = self.calcMarginLoss()
        loss = td_loss + self.margin_weight * margin_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), td_error

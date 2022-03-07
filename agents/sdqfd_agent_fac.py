from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from agents.dqn_agent_fac import DQNAgentFac
from utils import torch_utils

class SDQfDFac(DQNAgentFac):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1,
                 l=0.1, w=0.1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.margin_l = l
        self.margin_weight = w

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        q_p, q_dxy, q_dz, q_dtheta = self.loss_calc_dict['q_output']
        q_p_pred, q_dxy_pred, q_dz_pred, q_dtheta_pred = self.loss_calc_dict['q_pred']

        if is_experts.sum() == 0:
            return torch.tensor(0)
        batch_size = q_p.size(0)
        q_dxy = q_dxy.reshape(batch_size, -1)

        margin_losses = []
        for j in range(batch_size):
            if not is_experts[j]:
                margin_losses.append(torch.tensor(0).float().to(q_p.device))
                continue
            for i in range(4):
                if i == 0:
                    q_all = q_p[j]
                    qe = q_p_pred[j]
                    ae_id = p_id[j]
                elif i == 1:
                    q_all = q_dxy[j]
                    qe = q_dxy_pred[j]
                    ae_id = dxy_id[j]
                elif i == 2:
                    q_all = q_dz[j]
                    qe = q_dz_pred[j]
                    ae_id = dz_id[j]
                else:
                    q_all = q_dtheta[j]
                    qe = q_dtheta_pred[j]
                    ae_id = dtheta_id[j]

                over = q_all[(q_all > qe - self.margin_l) * (torch.arange(0, q_all.shape[0]).to(self.device) != ae_id)]
                if over.shape[0] == 0:
                    margin_losses.append(torch.tensor(0).float().to(self.device))
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

        self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return loss.item(), td_error

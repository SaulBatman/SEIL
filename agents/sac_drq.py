from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from networks.curl_sac_net import CURL
from utils.torch_utils import DrQAugment

class SACDrQ(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi / 16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False, obs_type='pixel'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, obs_type)
        self.K = 2
        self.M = 2
        self.aug_type = 'cn'

    def calcActorLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_expert = self._loadLossCalcDict()
        pis = []
        means = []
        log_pis = []
        actions = []
        is_experts = []
        min_qf_pis = []
        for _ in range(self.M):
            aug_obss = []
            aug_actions = []
            for i in range(batch_size):
                aug_obs, aug_action = DrQAugment(obs[i, 0].cpu().numpy(), action[i].cpu().numpy(), aug_type=self.aug_type)
                aug_obs = torch.tensor(aug_obs.reshape(1, 1, *aug_obs.shape)).to(self.device)
                aug_obs = torch.cat([aug_obs, obs[i:i+1, 1:2]], dim=1)
                aug_obss.append(aug_obs)
                aug_actions.append(aug_action)
            aug_obss = torch.cat(aug_obss, dim=0).to(self.device)
            aug_actions = torch.tensor(aug_actions).to(self.device)
            pi, log_pi, mean = self.actor.sample(aug_obss)
            qf1_pi, qf2_pi = self.critic(obs, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            pis.append(pi)
            means.append(mean)
            log_pis.append(log_pi)
            actions.append(aug_actions)
            is_experts.append(is_expert)
            min_qf_pis.append(min_qf_pi)
        pis = torch.cat(pis)
        means = torch.cat(means)
        log_pis = torch.cat(log_pis)
        actions = torch.cat(actions)
        is_experts = torch.cat(is_experts)
        min_qf_pis = torch.cat(min_qf_pis)

        self.loss_calc_dict['pi'] = pis
        self.loss_calc_dict['mean'] = means
        self.loss_calc_dict['log_pi'] = log_pis
        self.loss_calc_dict['action_idx'] = actions
        self.loss_calc_dict['is_experts'] = is_experts

        policy_loss = ((self.alpha * log_pis) - min_qf_pis).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        return policy_loss

    def calcCriticLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            next_q_values = []
            for _ in range(self.K):
                aug_next_obss = []
                for i in range(batch_size):
                    aug_next_obs, _ = DrQAugment(next_obs[i, 0].cpu().numpy(), action_idx=None, aug_type=self.aug_type)
                    aug_next_obs = torch.tensor(aug_next_obs.reshape(1, 1, *aug_next_obs.shape)).to(self.device)
                    aug_next_obs = torch.cat([aug_next_obs, next_obs[i:i+1, 1:2]], dim=1)
                    aug_next_obss.append(aug_next_obs)
                aug_next_obss = torch.cat(aug_next_obss, dim=0)
                next_state_action, next_state_log_pi, _ = self.actor.sample(aug_next_obss)
                next_state_log_pi = next_state_log_pi.reshape(batch_size)
                qf1_next_target, qf2_next_target = self.critic_target(aug_next_obss, next_state_action)
                qf1_next_target = qf1_next_target.reshape(batch_size)
                qf2_next_target = qf2_next_target.reshape(batch_size)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target
                next_q_values.append(next_q_value)
            next_q_values = torch.stack(next_q_values).mean(dim=0)
        qf1s = []
        qf2s = []
        for _ in range(self.M):
            aug_obss = []
            aug_actions = []
            for i in range(batch_size):
                aug_obs, aug_action = DrQAugment(obs[i, 0].cpu().numpy(), action[i].cpu().numpy(), aug_type=self.aug_type)
                aug_obs = torch.tensor(aug_obs.reshape(1, 1, *aug_obs.shape)).to(self.device)
                aug_obs = torch.cat([aug_obs, obs[i:i+1, 1:2]], dim=1)
                aug_obss.append(aug_obs)
                aug_actions.append(aug_action)
            aug_obss = torch.cat(aug_obss, dim=0).to(self.device)
            aug_actions = torch.tensor(aug_actions).to(self.device)
            qf1, qf2 = self.critic(aug_obss, aug_actions)
            qf1 = qf1.reshape(batch_size)
            qf2 = qf2.reshape(batch_size)
            qf1s.append(qf1)
            qf2s.append(qf2)
        qf1s = torch.cat(qf1s)
        qf2s = torch.cat(qf2s)
        next_q_values = next_q_values.repeat(self.M)

        qf1_loss = F.mse_loss(qf1s, next_q_values)
        qf2_loss = F.mse_loss(qf2s, next_q_values)
        with torch.no_grad():
            td_error = (0.5 * (torch.abs(qf2s - next_q_values) + torch.abs(qf1s - next_q_values))).reshape(batch_size, -1).mean(dim=1)
        return qf1_loss, qf2_loss, td_error

from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class SACfD(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 demon_w=0.1, demon_l='pi'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning)
        self.demon_w = demon_w
        assert demon_l in ['mean', 'pi']
        self.demon_l = demon_l

    def updateActorAndAlpha(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pi, log_pi, mean = self.actor.sample(obs)

        qf1_pi, qf2_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # add expert loss
        if is_experts.sum():
            if self.demon_l == 'pi':
                demon_loss = F.mse_loss(pi[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss
            else:
                demon_loss = F.mse_loss(mean[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return policy_loss, alpha_loss, alpha_tlogs


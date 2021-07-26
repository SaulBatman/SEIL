from agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class DDPG(A2CBase):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            next_action = self.forwardActor(next_states, next_obs, target_net=True)
            q_prime = self.forwardCritic(next_states, next_obs, next_action, target_net=True).reshape(batch_size)
            q_target_td = rewards + self.gamma * q_prime * non_final_masks

        self.critic_optimizer.zero_grad()
        q_output = self.forwardCritic(states, obs, action).reshape(batch_size)
        critic_loss = F.mse_loss(q_output, q_target_td)
        critic_loss.backward()
        self.critic_optimizer.step()

        # for p in self.critic.parameters():
        #     p.requires_grad = False

        self.actor_optimizer.zero_grad()
        pred_action = self.forwardActor(states, obs)
        actor_loss = (-self.forwardCritic(states, obs, pred_action)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # for p in self.critic.parameters():
        #     p.requires_grad = True

        self.targetSoftUpdate()

        with torch.no_grad():
            td_error = torch.abs(q_output - q_target_td)
        return (actor_loss.item(), critic_loss.item()), td_error

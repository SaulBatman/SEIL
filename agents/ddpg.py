from agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.parameters import heightmap_size, crop_size
from utils.torch_utils import centerCrop
from utils.schedules import LinearSchedule

class DDPG(A2CBase):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.std_schedule = LinearSchedule(1000, 0.1, 1)
        self.step = 0
        self.std_clip = 0.3

    def initNetwork(self, actor, critic, initialize_target=True):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


    def getDDPGAction(self, state, obs, evaluate):
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            obs = torch.cat([obs, state_tile], dim=1).to(self.device)
            if heightmap_size > crop_size:
                obs = centerCrop(obs, out=crop_size)
            std = self.std_schedule.value(self.step)
            dist = self.actor(obs, std)
            if evaluate:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

    def getEGreedyActions(self, state, obs, eps):
        return self.getDDPGAction(state, obs, evaluate=False)

    def getGreedyActions(self, state, obs):
        return self.getDDPGAction(state, obs, evaluate=True)

    def calcCriticLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, discounts = self._loadLossCalcDict()
        with torch.no_grad():
            std = self.std_schedule.value(self.step)
            dist = self.actor(next_obs, std)
            next_action = dist.sample(clip=self.std_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = rewards.unsqueeze(1) + (discounts.unsqueeze(1) * target_V)
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        with torch.no_grad():
            td_error = 0.5 * (torch.abs(Q1 - target_Q) + torch.abs(Q2 - target_Q))
        return critic_loss, td_error

    def calcActorLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, discounts = self._loadLossCalcDict()
        std = self.std_schedule.value(self.step)
        dist = self.actor(obs, std)
        action = dist.sample(clip=self.std_clip)
        self.loss_calc_dict['pi'] = action
        self.loss_calc_dict['mean'] = dist.mean
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()
        return actor_loss

    def updateActor(self):
        policy_loss = self.calcActorLoss()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        return policy_loss

    def updateCritic(self):
        critic_loss, td_error = self.calcCriticLoss()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        critic_loss, td_error = self.updateCritic()
        actor_loss = self.updateActor()
        self.targetSoftUpdate()
        self.step += 1
        self.loss_calc_dict = {}
        return (actor_loss.item(), critic_loss.item()), td_error

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, discounts = super()._loadLossCalcDict()

        # stack state as the second channel of the obs
        obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
        next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)

        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, discounts

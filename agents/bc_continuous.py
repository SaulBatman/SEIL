from agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class BehaviorCloningContinuous(A2CBase):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a)

    def initNetwork(self, actor):
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.actor)
        self.optimizers.append(self.actor_optimizer)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        assert is_experts.all()
        pi = self.forwardActor(states, obs)

        policy_loss = F.mse_loss(pi.float(), action.float())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), torch.tensor(0.)

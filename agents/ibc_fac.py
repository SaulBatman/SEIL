from agents.ibc import ImplicitBehaviorCloning, DerivativeFreeOptimizer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

class ImplicitBehaviorCloningFactored(ImplicitBehaviorCloning):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, ibc_ts=512, ibc_is=2048):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a)
        self.stochastic_optimizer_equ = DerivativeFreeOptimizer(device, n_a=2, train_samples=ibc_ts, inference_samples=ibc_is)
        self.stochastic_optimizer_inv = DerivativeFreeOptimizer(device, n_a=3, train_samples=ibc_ts, inference_samples=ibc_is)

    def forwardNetwork(self, state, obs, action, to_cpu=False):
        actor = self.actor
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)

        equ_output, inv_output = actor(stacked.to(self.device), action.to(self.device))
        equ_output = equ_output.reshape(state.shape[0], -1)
        inv_output = inv_output.reshape(state.shape[0], -1)
        if to_cpu:
            equ_output = equ_output.cpu()
            inv_output = inv_output.cpu()
        return equ_output, inv_output

    def getEGreedyActions(self, state, obs, eps):
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)

        with torch.no_grad():
            unscaled_actions_equ = self.stochastic_optimizer_equ.infer(stacked.to(self.device), self.actor.forwardEqu)
            unscaled_actions_inv = self.stochastic_optimizer_inv.infer(stacked.to(self.device), self.actor.forwardInv)
            unscaled_actions = torch.cat([unscaled_actions_inv[:, 0:1], unscaled_actions_equ, unscaled_actions_inv[:, 1:]], dim=1)

        return self.decodeActions(*[unscaled_actions[:, i] for i in range(self.n_a)])

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        assert is_experts.all()

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(batch_size, self.actor)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy_equ, energy_inv = self.forwardNetwork(states, obs, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits_equ = -1.0 * energy_equ
        logits_inv = -1.0 * energy_inv
        loss = F.cross_entropy(logits_equ, ground_truth) + F.cross_entropy(logits_inv, ground_truth)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.scheduler.step()

        return loss.item(), torch.tensor(0.)


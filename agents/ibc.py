from agents.a2c_base import A2CBase
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""
    def __init__(self, device, n_a=5, train_samples=512, inference_samples=2048, boundary_buffer=0.05):
        self.noise_scale = 0.33
        self.noise_shrink = 0.5
        self.iters = 3
        self.device = device
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.boundary_buffer = boundary_buffer
        self.bounds = np.array([[-1.0 for _ in range(n_a)], [1.0 for _ in range(n_a)]])
        action_range = self.bounds[1, :] - self.bounds[0, :]
        self.bounds[1, :] += action_range * self.boundary_buffer
        self.bounds[0, :] -= action_range * self.boundary_buffer

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        bounds = torch.as_tensor(self.bounds).to(self.device)

        samples = self._sample(x.size(0) * self.inference_samples)
        samples = samples.reshape(x.size(0), self.inference_samples, -1)

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]

class ImplicitBehaviorCloning(A2CBase):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, ibc_ts=512, ibc_is=2048):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a)
        self.stochastic_optimizer = DerivativeFreeOptimizer(device, n_a=n_a, train_samples=ibc_ts, inference_samples=ibc_is)

    def forwardNetwork(self, state, obs, action, to_cpu=False):
        actor = self.actor
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)

        output = actor(stacked.to(self.device), action.to(self.device))
        output = output.reshape(state.shape[0], -1)
        if to_cpu:
            output = output.cpu()
        return output

    def initNetwork(self, actor):
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.99)
        self.networks.append(self.actor)
        self.optimizers.append(self.actor_optimizer)

    def getEGreedyActions(self, state, obs, eps):
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        with torch.no_grad():
            unscaled_actions = self.stochastic_optimizer.infer(stacked.to(self.device), self.actor)

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
        energy = self.forwardNetwork(states, obs, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.scheduler.step()

        return loss.item(), torch.tensor(0.)


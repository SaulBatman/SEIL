from agents.dqn_agent_com import DQNAgentCom
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from networks.curl_sac_net import CURL
from utils.torch_utils import randomCrop, centerCrop

class CURLDQNCom(DQNAgentCom):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1, z_dim=50, crop_size=64):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.z_dim = z_dim
        self.crop_size = crop_size
        self.encoder_optimizer = None
        self.curl = None
        self.curl_optimizer = None

        self.encoder_tau = 0.05

    def encoderTargetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(
                self.target_net.encoder.parameters(), self.policy_net.encoder.parameters()
        ):
            t_param.data.copy_(self.encoder_tau * l_param.data + (1.0 - self.encoder_tau) * t_param.data)

    def targetSoftUpdate(self):
        super().targetSoftUpdate()
        self.encoderTargetSoftUpdate()

    def initNetwork(self, network, initialize_target=True):
        self.policy_net = network
        self.target_net = deepcopy(network)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.encoder_optimizer = torch.optim.Adam(self.policy_net.encoder.parameters(), lr=self.lr)

        self.curl = CURL(self.z_dim, self.policy_net.encoder, self.target_net.encoder).to(self.device)
        self.curl_optimizer = torch.optim.Adam(self.curl.parameters(), lr=self.lr)

        self.target_networks.append(self.target_net)

        self.networks.append(self.policy_net)
        self.networks.append(self.curl)

        self.optimizers.append(self.optimizer)
        self.optimizers.append(self.encoder_optimizer)
        self.optimizers.append(self.curl_optimizer)

    def updateCURL(self, update_target=False):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        obs_anchor = randomCrop(obs, out=self.crop_size)
        obs_pos = randomCrop(obs, out=self.crop_size)

        state_tile = states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, self.crop_size, self.crop_size)
        obs_anchor = torch.cat([obs_anchor, state_tile], dim=1)
        obs_pos = torch.cat([obs_pos, state_tile], dim=1)

        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos, ema=True)

        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = F.cross_entropy(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.curl_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.curl_optimizer.step()

        if update_target:
            self.encoderTargetSoftUpdate()

        return loss

    def getEGreedyActions(self, state, obs, eps):
        obs = centerCrop(obs, out=self.crop_size)
        return super().getEGreedyActions(state, obs, eps)

    def forwardNetworkDetachEncoder(self, state, obs, target_net=False, to_cpu=False):
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q = net(stacked.to(self.device), detach_encoder=True)
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q

    def updateCURLOnly(self, batch):
        self._loadBatchToDevice(batch)
        curl_loss = self.updateCURL(update_target=True)
        return 0, curl_loss.item()

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        next_obs = randomCrop(next_obs, out=self.crop_size)
        obs = randomCrop(obs, out=self.crop_size)
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size, -1).max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q = self.forwardNetworkDetachEncoder(states, obs)
        q_pred = q[torch.arange(batch_size), dxy_id, dz_id, dtheta_id, p_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.targetSoftUpdate()

        curl_loss = self.updateCURL(update_target=False)

        self.loss_calc_dict = {}


        return (td_loss.item(), curl_loss.item()), td_error
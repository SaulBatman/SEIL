from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.01, dy=0.01, dz=0.01, dr=np.pi/32):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        self.loss_calc_dict = {}

        self.p_range = torch.tensor([0, 1])
        self.dxy_range = torch.tensor([[0, 0],
                                       [-0.005, -0.005], [-0.005, 0], [-0.005, 0.005],
                                       [0, -0.005], [0, 0.005],
                                       [0.005, -0.005], [0.005, 0], [0.005, 0.005]])
        self.dz_range = torch.tensor([-0.005, 0, 0.005])
        self.d_theta_range = torch.tensor([0])

    def initNetwork(self, network):
        self.policy_net = network
        self.target_net = deepcopy(network)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-5)

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q_p, q_dxy, q_dz, q_dtheta = net(stacked.to(self.device))
        if to_cpu:
            q_p = q_p.to('cpu')
            q_dxy = q_dxy.to('cpu')
            q_dz = q_dz.to('cpu')
            q_dtheta = q_dtheta.to('cpu')
        return q_p, q_dxy, q_dz, q_dtheta

    def decodeActions(self, p_id, dxy_id, dz_id, dtheta_id):
        p = self.p_range[p_id]
        dxy = self.dxy_range[dxy_id]
        dz = self.dz_range[dz_id]
        dtheta = self.d_theta_range[dtheta_id]
        actions = torch.stack([p, dxy[:, 0], dxy[:, 1], dz, dtheta], dim=1)
        action_idxes = torch.stack([p_id, dxy_id, dz_id, dtheta_id], dim=1)
        return action_idxes, actions

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(state, obs, to_cpu=True)
            p_id = torch.argmax(q_p, 1)
            dxy_id = torch.argmax(q_dxy, 1)
            dz_id = torch.argmax(q_dz, 1)
            dtheta_id = torch.argmax(q_dtheta, 1)

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_p = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_p.size(1))
        p_id[rand_mask] = rand_p.long()
        rand_dxy = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dxy.size(1))
        dxy_id[rand_mask] = rand_dxy.long()
        rand_dz = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dz.size(1))
        dz_id[rand_mask] = rand_dz.long()
        rand_dtheta = torch.randint_like(torch.empty(rand_mask.sum()), 0, q_dtheta.size(1))
        dtheta_id[rand_mask] = rand_dtheta.long()
        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def getActionFromPlan(self, plan):
        primitive = plan[:, 0:1]
        dxy = plan[:, 1:3]
        dz = plan[:, 3:4]
        dr = plan[:, 4:5]

        p_id = torch.argmin(torch.abs(self.p_range - primitive), 1)
        dxy_id = torch.argmin((dxy.unsqueeze(1) - self.dxy_range).abs().sum(2), 1)
        dz_id = torch.argmin(torch.abs(self.dz_range - dz), 1)
        dtheta_id = torch.argmin(torch.abs(self.d_theta_range - dr), 1)

        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_p_prime, q_dxy_prime, q_dz_prime, q_dtheta_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_p_prime.max(1)[0] + q_dxy_prime.max(1)[0] + q_dz_prime.max(1)[0] + q_dtheta_prime.max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q_p, q_dxy, q_dz, q_dtheta = self.forwardNetwork(states, obs)
        q_p_pred = q_p[torch.arange(batch_size), p_id]
        q_dxy_pred = q_dxy[torch.arange(batch_size), dxy_id]
        q_dz_pred = q_dz[torch.arange(batch_size), dz_id]
        q_dtheta_pred = q_dtheta[torch.arange(batch_size), dtheta_id]
        q_pred = q_p_pred + q_dxy_pred + q_dz_pred + q_dtheta_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error


    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = image_tensor
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, image_tensor, xy_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def train(self):
        self.policy_net.train()

    def getModelStr(self):
        return str(self.policy_net)

    def updateTarget(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def saveModel(self, path_pre):
        torch.save(self.policy_net.state_dict(), '{}_network.pt'.format(path_pre))

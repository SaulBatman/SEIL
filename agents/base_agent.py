from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class BaseAgent:
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1):
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
        self.p_range = torch.tensor([1])
        if n_p == 2:
            self.p_range = torch.tensor([0, 1])

        self.d_theta_range = torch.tensor([0])
        if n_theta == 3:
            self.d_theta_range = torch.tensor([-dr, 0, dr])

        self.dxy_range = torch.tensor([[-dx, -dy], [-dx, 0], [-dx, dy],
                                       [0, -dy], [0, 0], [0, dy],
                                       [dx, -dy], [dx, 0], [dx, dy]])
        self.dz_range = torch.tensor([-dz, 0, dz])

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        raise NotImplementedError

    def getEGreedyActions(self, state, obs, eps):
        raise NotImplementedError

    def calcTDLoss(self):
        raise NotImplementedError

    def initNetwork(self, network):
        self.policy_net = network
        self.target_net = deepcopy(network)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-5)

    def decodeActions(self, p_id, dxy_id, dz_id, dtheta_id):
        p = self.p_range[p_id]
        dxy = self.dxy_range[dxy_id]
        dz = self.dz_range[dz_id]
        dtheta = self.d_theta_range[dtheta_id]
        actions = torch.stack([p, dxy[:, 0], dxy[:, 1], dz, dtheta], dim=1)
        action_idxes = torch.stack([p_id, dxy_id, dz_id, dtheta_id], dim=1)
        return action_idxes, actions

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
        self.target_net.train()

    def getModelStr(self):
        return str(self.policy_net)

    def updateTarget(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def saveModel(self, path_pre):
        torch.save(self.policy_net.state_dict(), '{}_network.pt'.format(path_pre))

    def loadModel(self, path_pre):
        path = '{}_network.pt'.format(path_pre)
        print('loading {}'.format(path))
        self.policy_net.load_state_dict(torch.load(path))
        self.updateTarget()

    def getSaveState(self):
        state = {}
        state['policy_net'] = self.policy_net.state_dict()
        state['target_net'] = self.target_net.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        return state

    def loadFromState(self, save_state):
        self.policy_net.load_state_dict(save_state['policy_net'])
        self.target_net.load_state_dict(save_state['target_net'])
        self.optimizer.load_state_dict(save_state['optimizer'])
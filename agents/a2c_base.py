from agents.base_agent import BaseAgent
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class A2CBase(BaseAgent):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001):
        super().__init__(lr, gamma, device, dx, dy, dz, dr)
        self.tau = tau

        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None

        self.p_range = torch.tensor([0, 1])
        self.dtheta_range = torch.tensor([-dr, dr])
        self.dx_range = torch.tensor([-dx, dx])
        self.dy_range = torch.tensor([-dy, dy])
        self.dz_range = torch.tensor([-dz, dz])

        self.n_a = n_a

    def targetHardUpdate(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


    def initNetwork(self, actor, critic, initialize_target=True):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0], weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1], weight_decay=1e-5)
        if initialize_target:
            self.actor_target = deepcopy(actor)
            self.critic_target = deepcopy(critic)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.target_networks.append(self.actor_target)
        self.target_networks.append(self.critic_target)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def forwardActor(self, state, obs, target_net=False, to_cpu=False):
        actor = self.actor if not target_net else self.actor_target
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        output = actor(stacked.to(self.device))
        if to_cpu:
            output = output.cpu()
        return output

    def forwardCritic(self, state, obs, action, target_net=False, to_cpu=False):
        critic = self.critic if not target_net else self.critic_target
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        output = critic(stacked, action)
        if to_cpu:
            output = output.cpu()
        return output

    def decodeActions(self, *args):
        unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz = args[0], args[1], args[2], args[3]

        p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
        dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
        dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
        dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

        if self.n_a == 5:
            unscaled_dtheta = args[4]
            dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
            actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)
        else:
            actions = torch.stack([p, dx, dy, dz], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1)

        return unscaled_actions, actions

    def decodeSingleActions(self, *args):
        unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz = args[0], args[1], args[2], args[3]

        p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
        dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
        dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
        dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

        if self.n_a == 5:
            unscaled_dtheta = args[4]
            dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
            actions = torch.stack([p, dx, dy, dz, dtheta], dim=0)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=0)
        else:
            actions = torch.stack([p, dx, dy, dz], dim=0)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=0)

        return unscaled_actions, actions

    def getActionFromPlan(self, plan):
        def getUnscaledAction(action, action_range):
            unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
            return unscaled_action
        dx = plan[:, 1].clamp(*self.dx_range)
        p = plan[:, 0].clamp(*self.p_range)
        dy = plan[:, 2].clamp(*self.dy_range)
        dz = plan[:, 3].clamp(*self.dz_range)

        unscaled_p = getUnscaledAction(p, self.p_range)
        unscaled_dx = getUnscaledAction(dx, self.dx_range)
        unscaled_dy = getUnscaledAction(dy, self.dy_range)
        unscaled_dz = getUnscaledAction(dz, self.dz_range)

        if self.n_a == 5:
            dtheta = plan[:, 4].clamp(*self.dtheta_range)
            unscaled_dtheta = getUnscaledAction(dtheta, self.dtheta_range)
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta)
        else:
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz)

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            unscaled_actions = self.forwardActor(state, obs, to_cpu=True)

        rand = torch.tensor(np.random.uniform(0, 1, state.size(0)))
        rand_mask = rand < eps
        rand_act = 2*torch.rand(rand_mask.sum(), self.n_a)-1
        unscaled_actions[rand_mask] = rand_act
        return self.decodeActions(*[unscaled_actions[:, i] for i in range(self.n_a)])

    # def getExpertActions(self, state, obs, eps):
    #     with torch.no_grad():
    #         unscaled_actions = self.forwardActor(state, obs, to_cpu=True)
    #     return self.decodeActions(*[unscaled_actions[:, i] for i in range(self.n_a)])

    def getInvBCActions(self, scaled_action0, scaled_action1, sigma, method='gaussian'):
        if method == 'gaussian':
            p_inv = scaled_action0[0]
            # sigma = 0.2 # 20% disturbance
            x_inv = np.clip(np.random.normal(-scaled_action1[1], sigma), -1, 1)
            y_inv = np.clip(np.random.normal(-scaled_action1[2], sigma), -1, 1)
            z_inv = np.clip(np.random.normal(-scaled_action1[3], sigma), -1, 1)
            r_inv = np.clip(np.random.normal(-scaled_action1[4], sigma), -1, 1)
        elif method == 'identical':
            p_inv = scaled_action0[0]
            x_inv = -scaled_action1[1]
            y_inv = -scaled_action1[2]
            z_inv = -scaled_action1[3]
            r_inv = -scaled_action1[4]
        elif method == 'uniform':
            pass
        
        # return self.decodeActions(*[scaled_actions_new[:, i] for i in range(self.n_a)])
        unscaled_actions = torch.tensor([[p_inv, x_inv, y_inv, z_inv, r_inv]])
        return self.decodeActions(*[unscaled_actions[:, i] for i in range(self.n_a)])

    def getSimBCActions(self, scaled_action_inv, star_gripper_action):
        scaled_actions_new = -scaled_action_inv.clone().detach()
        scaled_actions_new[0][0] = star_gripper_action
        # return self.decodeActions(*[scaled_actions_new[:, i] for i in range(self.n_a)])
        return self.decodeActions(*[scaled_actions_new[:, i] for i in range(self.n_a)])


    def updateTarget(self):
        pass
import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import torch_utils

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACEncoder(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        if obs_shape[1] == 128:
            self.conv = torch.nn.Sequential(
                # 128x128
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 64x64
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 32x32
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 16x16
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 8x8
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Flatten(),
                torch.nn.Linear(512 * 8 * 8, out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                # 64x64
                nn.Conv2d(obs_shape[0], 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 32x32
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 16x16
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # 8x8
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Flatten(),
                torch.nn.Linear(512 * 8 * 8, out_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class SACCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = SACEncoder(obs_shape, 1024)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.conv = SACEncoder(obs_shape, 1024)

        self.mean_linear = nn.Linear(1024, action_dim)
        self.log_std_linear = nn.Linear(1024, action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.conv = SACEncoder(obs_shape, 1024)
        self.mean = nn.Linear(1024, action_dim)
        self.noise = torch.Tensor(action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, x):
        mean = self.forward(x)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class SACVecCritic(nn.Module):
    def __init__(self, obs_dim=7, action_dim=5):
        super().__init__()
        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+action_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+action_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )
        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        out_1 = self.critic_fc_1(torch.cat((obs, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((obs, act), dim=1))
        return out_1, out_2

class SACVecGaussianPolicy(nn.Module):
    def __init__(self, obs_dim=7, action_dim=5, action_space=None):
        super().__init__()
        self.action_dim = action_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 2 * action_dim)
        )
        # self.mean_linear = nn.Linear(1024, action_dim)
        # self.log_std_linear = nn.Linear(1024, action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        out = self.fc(x)
        mean = out[:, :self.action_dim]
        log_std = out[:, self.action_dim:]
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from utils import torch_utils

from e2cnn import gspaces
from e2cnn import nn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def tieWeights(src, trg):
    assert type(src) == type(trg)
    trg.weights = src.weights
    trg.bias = src.bias

class CURLEquiSACEncoder(torch.nn.Module):
    def __init__(self, input_shape=(2, 128, 128), output_dim=50, initialize=True, N=4):
        super().__init__()
        obs_channel = input_shape[0]
        n_out = output_dim
        self.obs_channel = input_shape[0]
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

        self.fc = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
        # self.ln = torch.nn.LayerNorm(output_dim*N)


    def forward(self, x, detach=False):
        batch_size = x.shape[0]
        obs_geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        # h = self.conv(obs_geo)
        if detach:
            # h = h.detach()
            with torch.no_grad():
                h = self.conv(obs_geo)
        else:
            h = self.conv(obs_geo)
        h_fc = self.fc(h)
        h_fc = h_fc.tensor.reshape(batch_size, -1)
        # h_fc = self.ln(h_fc)
        return h_fc

    def copyConvWeightsFrom(self, source):
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.R2Conv):
                tieWeights(src=source.conv[i], trg=self.conv[i])

class CURLEquiSACCritic(torch.nn.Module):
    def __init__(self, encoder, encoder_output_dim=50, action_dim=5, initialize=True, N=4):
        super().__init__()
        self.encoder = encoder
        self.n_hidden = encoder_output_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1

        # Q1
        self.q1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr] + (action_dim - 2) * [
                self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        # Q2
        self.q2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr] + (action_dim - 2) * [
                self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act, detach_encoder=False):
        batch_size = obs.shape[0]
        enc_out = self.encoder(obs, detach=detach_encoder)
        enc_out = enc_out.reshape(batch_size, -1, 1, 1)
        enc_out_geo = nn.GeometricTensor(enc_out, nn.FieldType(self.c4_act, self.n_hidden*[self.c4_act.regular_repr]))
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((enc_out_geo.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.q1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.q2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class CURLEquiSACGaussianPolicy(torch.nn.Module):
    def __init__(self, encoder, encoder_output_dim=50, action_dim=5, initialize=True, N=4):
        super().__init__()
        self.encoder = encoder

        self.n_hidden = encoder_output_dim
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1

        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, encoder_output_dim * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, obs, detach_encoder=False):
        batch_size = obs.shape[0]
        enc_out = self.encoder(obs, detach=detach_encoder)
        enc_out = enc_out.reshape(batch_size, -1, 1, 1)
        enc_out_geo = nn.GeometricTensor(enc_out, nn.FieldType(self.c4_act, self.n_hidden*[self.c4_act.regular_repr]))
        conv_out = self.conv(enc_out_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x, detach_encoder=False):
        mean, log_std = self.forward(x, detach_encoder=detach_encoder)
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

encoder = CURLEquiSACEncoder(output_dim=64, initialize=False)
actor = CURLEquiSACGaussianPolicy(encoder, encoder_output_dim=64, initialize=False)
critic = CURLEquiSACCritic(encoder, encoder_output_dim=64, initialize=False)

obs = torch.zeros(1, 2, 128, 128)
action = torch.zeros(1, 5)

action_sample = actor.sample(obs)
q = critic(obs, action)

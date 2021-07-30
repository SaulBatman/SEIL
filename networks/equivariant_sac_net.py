import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class EquivariantEncoder(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
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

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)



class EquivariantSACCritic(torch.nn.Module):
    def __init__(self, obs_channel=2, action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.img_conv = EquivariantEncoder(obs_channel, n_hidden, initialize, N)

        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACActor(torch.nn.Module):
    def __init__(self, obs_channel=2, action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            EquivariantEncoder(obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    critic = EquivariantSACCritic(obs_channel=2, action_dim=4, n_hidden=64, initialize=False)
    o = torch.zeros(1, 2, 128, 128)
    o[0, 0, 10:20, 10:20] = 1
    a = torch.zeros(1, 4)
    a[0, 1:3] = torch.tensor([-1., -1.])

    o2 = torch.rot90(o, 1, [2, 3])
    a2 = torch.zeros(1, 4)
    a2[0, 1:3] = torch.tensor([1., -1.])

    out = critic(o, a)

    actor = EquivariantSACActor(obs_channel=2, action_dim=4, n_hidden=64, initialize=False)
    out2 = actor(o)


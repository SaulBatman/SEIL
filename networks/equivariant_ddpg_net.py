import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn

class EquivariantDDPGCritic(torch.nn.Module):
    def __init__(self, action_dim=5, initialize=True):
        super().__init__()
        self.c4_act = gspaces.Rot2dOnR2(4)
        self.img_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, 2 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
            # nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
            #           nn.FieldType(self.c4_act, 256 * [self.c4_act.trivial_repr]),
            #           kernel_size=1, padding=0, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.trivial_repr]), inplace=True),
        )

        self.critic = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, 2*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]))
        out = self.critic(cat_geo).tensor.reshape(batch_size, 1)
        return out

class EquivariantDDPGActor(torch.nn.Module):
    def __init__(self, action_dim=5, initialize=True):
        super().__init__()
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, 2 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
            # nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
            #           nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
            #           kernel_size=1, padding=0, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.trivial_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, 2*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        act = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        act = torch.tanh(act)
        return act


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    critic = EquivariantDDPGCritic(4, initialize=False)
    o = torch.zeros(1, 2, 128, 128)
    o[0, 0, 10:20, 10:20] = 1
    a = torch.zeros(1, 4)
    a[0, 1:3] = torch.tensor([-1., -1.])

    o2 = torch.rot90(o, 1, [2, 3])
    a2 = torch.zeros(1, 4)
    a2[0, 1:3] = torch.tensor([1., -1.])

    out = critic(o, a)

    actor = EquivariantDDPGActor(4, initialize=False)
    out2 = actor(o)


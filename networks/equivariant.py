import torch
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn
from networks.cnn_fcn import FCN, BasicBlock

# class EquivariantCNNFac(torch.nn.Module):
#     def __init__(self, initialize=True, n_p=2, n_theta=1):
#         super().__init__()
#         self.r2_act = gspaces.Rot2dOnR2(4)
#         self.conv = torch.nn.Sequential(
#             # 128x128
#             nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
#                       nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), 2),
#             # 64x64
#             nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
#             # 32x32
#             nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
#             # 16x16
#             nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
#             # 8x8
#             nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
#         )
#
#         self.dxy_layer = torch.nn.Sequential(
#             nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
#                       kernel_size=8, padding=0),
#             nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
#             nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
#                       nn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr]),
#                       kernel_size=1, padding=0),
#         )
#
#         self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]))
#         self.fc1 = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Linear(256*8*8, 1024),
#             torch.nn.ReLU(inplace=True),
#         )
#
#         self.p_layer = torch.nn.Sequential(
#             torch.nn.Linear(1024, n_p)
#         )
#
#         self.dz_layer = torch.nn.Sequential(
#             torch.nn.Linear(1024, 3)
#         )
#
#         self.dtheta_layer = torch.nn.Sequential(
#             torch.nn.Linear(1024, n_theta)
#         )
#
#         self.dxy0_layer = torch.nn.Sequential(
#             torch.nn.Linear(1024, 1)
#         )
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]))
#         h = self.conv(x)
#         dxy = self.dxy_layer(h).tensor.reshape(batch_size, 2, 4)
#         inv_h = self.fc1(self.group_pool(h).tensor)
#         dxy0 = self.dxy0_layer(inv_h)
#         dxy = torch.stack((dxy0[:, 0],
#                            dxy[:, 0, 0], dxy[:, 1, 0], dxy[:, 0, 1],
#                            dxy[:, 1, 3], dxy[:, 1, 1],
#                            dxy[:, 0, 3], dxy[:, 1, 2], dxy[:, 0, 2]),
#                           dim=1)
#         p = self.p_layer(inv_h)
#         dz = self.dz_layer(inv_h)
#         dtheta = self.dtheta_layer(inv_h)
#         return p, dxy, dz, dtheta

class EquivariantCNNFac(torch.nn.Module):
    def __init__(self, n_input_channel=2, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.c4_act = gspaces.Rot2dOnR2(4)
        self.c4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, n_input_channel * [self.c4_act.trivial_repr]),
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
                      nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]), 2),
            # 3x3
        )

        self.c4_33_out = nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                                   nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.c4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, 256 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_inv * [self.c4_act.trivial_repr] + self.n_theta * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.c4_act, 2 * [self.c4_act.trivial_repr]))
        h = self.c4_conv(x)
        dxy = self.c4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.c4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p::4]
        return p, dxy, dz, dtheta

class EquivariantResBlock(torch.nn.Module):
    def __init__(self, space, input_channel, hidden_channel, initialize=True):
        super().__init__()
        feat_type_in = nn.FieldType(space, input_channel * [space.regular_repr])
        feat_type_hid = nn.FieldType(space, hidden_channel * [space.regular_repr])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3, padding=1, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channel != hidden_channel:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3, padding=1, initialize=initialize),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquivariantCNNFacD4(torch.nn.Module):
    def __init__(self, n_input_channel=2, n_hidden=128, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, n_input_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_hidden//8 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden//8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden//8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden//8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden//4 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden//4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden//4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden//4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden//2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden//2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden//2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden//2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden*2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden*2 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden*2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), 2),
            # 3x3
        )

        self.d4_33_out = nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                                   nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.d4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, (self.n_inv+self.n_theta) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, self.n_input_channel * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p:]
        return p, dxy, dz, dtheta

class EquivariantResFacD4(torch.nn.Module):
    def __init__(self, n_input_channel=2, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, n_input_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), inplace=True),
            EquivariantResBlock(self.d4_act, 16, 16, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), 2),
            # 64x64
            EquivariantResBlock(self.d4_act, 16, 32, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]), 2),
            # 32x32
            EquivariantResBlock(self.d4_act, 32, 64, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]), 2),
            # 16x16
            EquivariantResBlock(self.d4_act, 64, 128, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]), 2),
            # 8x8
            EquivariantResBlock(self.d4_act, 128, 256, initialize),

            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), 2),
            # 3x3
        )

        self.d4_33_out = nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                                   nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.d4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, (self.n_inv+self.n_theta) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, self.n_input_channel * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p:]
        return p, dxy, dz, dtheta

class EquivariantCNNFacD45x5(torch.nn.Module):
    def __init__(self, n_input_channel=2, initialize=True, n_p=2, n_theta=1, n_z=3):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.n_theta = n_theta
        self.n_p = n_p
        self.n_z = n_z
        self.n_inv = n_theta + n_p + n_z

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, n_input_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 14x14
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), 2),
            # 7x7
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 5x5
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
        )

        self.d4_33_out = nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                                   nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.d4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, self.n_inv * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, self.n_input_channel * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :self.n_z]
        p = inv_out[:, self.n_z:self.n_z+self.n_p]
        dtheta = inv_out[:, self.n_z+self.n_p:]
        return p, dxy, dz, dtheta

class EquiCNNFacD4WithNonEquiFCN(torch.nn.Module):
    def __init__(self, n_input_channel=2, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.fcn = FCN(n_input_channel-1, n_input_channel-1)

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, n_input_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, 16 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, 32 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, 64 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, 128 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), 2),
            # 3x3
        )

        self.d4_33_out = nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                                   nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.d4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, (self.n_inv+self.n_theta) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x[:, 0:self.n_input_channel-1] = self.fcn(x[:, 0:self.n_input_channel-1])
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, self.n_input_channel * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p:]
        return p, dxy, dz, dtheta

class EquiCNNFacD4WithNonEquiEnc(torch.nn.Module):
    def __init__(self, n_input_channel=2, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.non_equi_enc = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1, ),
            torch.nn.ReLU(inplace=True),
            BasicBlock(32, 32, dilation=1),

            torch.nn.MaxPool2d(2),
            # 64x64
            BasicBlock(32, 64, downsample=torch.nn.Conv2d(32, 64, kernel_size=1, bias=False), dilation=1),

            torch.nn.MaxPool2d(2),
            # 32x32
            BasicBlock(64, 128, downsample=torch.nn.Conv2d(64, 128, kernel_size=1, bias=False), dilation=1),

            torch.nn.MaxPool2d(2),
            # 16x16
            BasicBlock(128, 256, downsample=torch.nn.Conv2d(128, 256, kernel_size=1, bias=False), dilation=1),

            torch.nn.MaxPool2d(2),
            # 8x8
            BasicBlock(256, 512, downsample=torch.nn.Conv2d(256, 512, kernel_size=1, bias=False), dilation=1),
        )

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, 512 * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 6x6
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), 2),
            # 3x3
        )

        self.d4_33_out = nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                                   nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                   kernel_size=1, padding=0, initialize=initialize)
        self.d4_11_out = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, 256 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, (self.n_inv+self.n_theta) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.non_equi_enc(x)
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, 512 * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p:]
        return p, dxy, dz, dtheta

# 3x3 out
class EquivariantCNNCom(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, self.n_inv * [self.r2_act.trivial_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        out = self.conv(x).tensor.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return out

# 1x1 out fill
class EquivariantCNNCom2(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.Rot2dOnR2(8)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),

            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            # 1x1
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, self.n_inv * [self.r2_act.trivial_repr] + self.n_inv * [self.r2_act.regular_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        dxy = self.conv(x)
        dxy_0 = dxy[:, :self.n_inv].tensor.reshape(batch_size, self.n_inv, 1)
        dxy_1 = dxy[:, self.n_inv:].tensor.reshape(batch_size, self.n_inv, 8)
        dxy = torch.stack((dxy_1[:, :, 0], dxy_1[:, :, 1], dxy_1[:, :, 2],
                           dxy_1[:, :, 7], dxy_0[:, :, 0], dxy_1[:, :, 3],
                           dxy_1[:, :, 6], dxy_1[:, :, 5], dxy_1[:, :, 4]),
                          dim=1)

        return dxy

# 3x3 out
class EquivariantCNNComD4(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.FlipRot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, self.n_inv * [self.r2_act.trivial_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        out = self.conv(x).tensor.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return out

if __name__ == '__main__':
    net = EquivariantResFacD4(initialize=False, n_p=2, n_theta=3)
    a = torch.rand((16, 2, 128, 128))
    out = net(a)
    print(1)
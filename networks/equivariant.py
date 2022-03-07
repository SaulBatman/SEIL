import torch
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn

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
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.c4_act = gspaces.Rot2dOnR2(4)
        self.c4_conv = torch.nn.Sequential(
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

class EquivariantCNNFacD4(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.d4_act = gspaces.FlipRot2dOnR2(4)
        self.d4_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, 2 * [self.d4_act.trivial_repr]),
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
        x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, 2 * [self.d4_act.trivial_repr]))
        h = self.d4_conv(x)
        dxy = self.d4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.d4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]
        dtheta = inv_out[:, 3+self.n_p:]
        return p, dxy, dz, dtheta

class EquivariantCNNFac3(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.c4_act = gspaces.Rot2dOnR2(4)
        self.c4_conv = torch.nn.Sequential(
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

        self.r_act = gspaces.Flip2dOnR2()
        self.r_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r_act, 2 * [self.r_act.trivial_repr]),
                      nn.FieldType(self.r_act, 16 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 16 * [self.r_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r_act, 16 * [self.r_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r_act, 16 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 32 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 32 * [self.r_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r_act, 32 * [self.r_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r_act, 32 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 64 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 64 * [self.r_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r_act, 64 * [self.r_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r_act, 64 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 128 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 128 * [self.r_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r_act, 128 * [self.r_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r_act, 128 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 1x1
            nn.ReLU(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.r_act, 256 * [self.r_act.regular_repr]),
                      nn.FieldType(self.r_act, 1 * [self.r_act.trivial_repr] + 1 * [self.r_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_c4 = nn.GeometricTensor(x, nn.FieldType(self.c4_act, 2 * [self.c4_act.trivial_repr]))
        h = self.c4_conv(x_c4)
        dxy = self.c4_33_out(h).tensor.reshape(batch_size, -1)
        inv_out = self.c4_11_out(h).tensor.reshape(batch_size, -1)
        dz = inv_out[:, :3]
        p = inv_out[:, 3:3+self.n_p]

        x_r = nn.GeometricTensor(x, nn.FieldType(self.r_act, 2 * [self.r_act.trivial_repr]))
        dtheta = self.r_conv(x_r).tensor.reshape(batch_size, -1)
        if self.n_theta == 3:
            dtheta = torch.stack((dtheta[:, 1], dtheta[:, 0], dtheta[:, 2]), dim=1)
        else:
            dtheta = dtheta[:, 0:1]
        return p, dxy, dz, dtheta

class EquivariantCNNFac2(torch.nn.Module):
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 1 + 3 + n_theta + n_p
        self.n_theta = n_theta
        self.n_p = n_p

        self.r2_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
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

            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
            # 1x1
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, self.n_inv * [self.r2_act.trivial_repr] + 2 * [self.r2_act.regular_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        out = self.conv(x)
        dxy0 = out[:, 0:1].tensor.reshape(batch_size, 1)
        dz = out[:, 1:4].tensor.reshape(batch_size, 3)
        dtheta = out[:, 4:4+self.n_theta].tensor.reshape(batch_size, self.n_theta)
        p = out[:, 4+self.n_theta:4+self.n_theta+self.n_p].tensor.reshape(batch_size, self.n_p)
        dxy = out[:, 4+self.n_theta+self.n_p:].tensor.reshape(batch_size, 2, 4)

        dxy = torch.stack((dxy[:, 0, 0], dxy[:, 1, 0], dxy[:, 0, 1],
                           dxy[:, 1, 3], dxy0[:, 0], dxy[:, 1, 1],
                           dxy[:, 0, 3], dxy[:, 1, 2], dxy[:, 0, 2]),
                          dim=1)
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
    net = EquivariantCNNFac3(initialize=False, n_p=2, n_theta=3)
    a = torch.rand((16, 2, 128, 128))
    out = net(a)
    print(1)
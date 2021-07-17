import torch
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn

class EquivariantCNN(torch.nn.Module):
    def __init__(self, initialize=True):
        super().__init__()
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
        )

        self.dxy_layer = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=8, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr]),
                      kernel_size=1, padding=0),
        )

        self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]))

        self.p_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 2)
        )

        self.dz_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 3)
        )

        self.dtheta_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )

        self.dxy0_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]))
        h = self.conv(x)
        dxy = self.dxy_layer(h).tensor.reshape(batch_size, 2, 4)
        inv_h = self.group_pool(h).tensor
        dxy0 = self.dxy0_layer(inv_h)
        dxy = torch.stack((dxy0[:, 0],
                           dxy[:, 0, 0],
                           dxy[:, 1, 0],
                           dxy[:, 0, 1],
                           dxy[:, 1, 1],
                           dxy[:, 1, 2],
                           dxy[:, 0, 2],
                           dxy[:, 1, 3],
                           dxy[:, 0, 3]), dim=1)
        p = self.p_layer(inv_h)
        dz = self.dz_layer(inv_h)
        dtheta = self.dtheta_layer(inv_h)
        return p, dxy, dz, dtheta

if __name__ == '__main__':
    net = EquivariantCNN(initialize=False)
    a = torch.rand((16, 2, 128, 128))
    out = net(a)
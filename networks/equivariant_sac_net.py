import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn

from networks.sac_networks import SACGaussianPolicyBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, initialize=True):
        super(EquiResBlock, self).__init__()
        r2_act = gspaces.Rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
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

class EquivariantEncoder128(torch.nn.Module):
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

class EquivariantEncoder128Dihedral(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.FlipRot2dOnR2(N)
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

class EquivariantEncoder128DihedralK5(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 14x14
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),
            # 12x12
            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 10x10
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 5x5
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=5, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128SO2_1(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//8 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//8 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//4 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//4 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 4 * [self.repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//2 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//2 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 2 * [self.repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * [self.repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out*2 * [self.repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out*2 * [self.repr])),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * [self.repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantSO2Layer(torch.nn.Module):
    def __init__(self, gs, r1, out_channel, kernel_size, padding, stride, initialize):
        super().__init__()
        self.c4_act = gs
        irreps = []
        for n, irr in self.c4_act.fibergroup.irreps.items():
            if n != self.c4_act.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        I = len(irreps)
        # S = nn.FieldType(self.c4_act, irreps).size + 1
        # M = S + I
        trivials = nn.FieldType(self.c4_act, [self.c4_act.trivial_repr] * out_channel)
        gates = nn.FieldType(self.c4_act, [self.c4_act.trivial_repr] * out_channel * I)
        gated = nn.FieldType(self.c4_act, irreps * out_channel).sorted()
        gate = gates + gated
        r2 = trivials + gate
        self.conv = nn.R2Conv(r1, r2, kernel_size=kernel_size, padding=padding, stride=stride, initialize=initialize)
        labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
        modules = [
            (nn.ELU(trivials), "trivial"),
            (nn.GatedNonLinearity1(gate), "gate")
        ]
        self.nnl = nn.MultipleModule(self.conv.out_type, labels, modules)

    def forward(self, x):
        return self.nnl(self.conv(x))

class EquivariantEncoder128SO2_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]

        self.conv = torch.nn.Sequential(
            # 128x128
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                                n_out//8, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//8 * self.repr).sorted(), 2),
            # 64x64
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out//8 * self.repr).sorted(),
                                n_out//4, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//4 * self.repr).sorted(), 2),
            # 32x32
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out // 4 * self.repr).sorted(),
                                n_out // 2, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 2 * self.repr).sorted(), 2),
            # 16x16
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out // 2 * self.repr).sorted(),
                                n_out, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * self.repr).sorted(), 2),
            # 8x8
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out * self.repr).sorted(),
                                n_out*2, kernel_size=3, padding=1, stride=1, initialize=initialize),
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out*2 * self.repr).sorted(),
                                n_out, kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * self.repr).sorted(), 2),
            # 3x3
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out * self.repr).sorted(),
                                n_out, kernel_size=3, padding=0, stride=1, initialize=initialize),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128NoPool(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 4),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(geo, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128Small(torch.nn.Module):
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
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 4),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 4),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            # 6x6
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 1x1
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(geo, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128Res(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), inplace=True),
            # 128x128
            EquiResBlock(n_out//8, n_out//8, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            EquiResBlock(n_out//8, n_out//4, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            EquiResBlock(n_out//4, n_out//2, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            EquiResBlock(n_out//2, n_out, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            EquiResBlock(n_out, n_out*2, 3, N, initialize),
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

class EquivariantEncoder64_1(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
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

class EquivariantEncoder64_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
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

def getEnc(obs_size, enc_id):
    assert obs_size in [128, 64]
    if obs_size == 128:
        if enc_id == 1:
            return EquivariantEncoder128
        elif enc_id == 2:
            return EquivariantEncoder128Res
        elif enc_id == 3:
            return EquivariantEncoder128NoPool
        elif enc_id == 4:
            return EquivariantEncoder128Small
        else:
            raise NotImplementedError
    else:
        if enc_id == 1:
            return EquivariantEncoder64_1
        else:
            return EquivariantEncoder64_2

class EquivariantSACCritic(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
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
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticDihedral(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
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
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticDihedralShareEnc(torch.nn.Module):
    def __init__(self, enc, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.img_conv = enc
        self.n_rho1 = 2 if N==2 else 1
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
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
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticSO2_1(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_1
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_hidden * [self.repr])),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * [self.repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_hidden * [self.repr])),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * [self.repr])),
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
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticSO2_2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.critic_1 = torch.nn.Sequential(
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_hidden * self.repr).sorted() + nn.FieldType(self.c4_act, (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                                n_hidden, kernel_size=1, padding=0, stride=1, initialize=initialize),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted()),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * 4 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_hidden * self.repr).sorted() + nn.FieldType(self.c4_act, (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                                n_hidden, kernel_size=1, padding=0, stride=1, initialize=initialize),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted()),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * 4 * [self.c4_act.trivial_repr]),
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
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * self.repr).sorted() + nn.FieldType(self.c4_act, n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticNoGP(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)

        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + 1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]), inplace=True),
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

class EquivariantSACActor(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.n_rho1 = 2 if N==2 else 1
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
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

class EquivariantSACActorDihedral(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
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

class EquivariantSACActorDihedralShareEnc(SACGaussianPolicyBase):
    def __init__(self, enc, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.enc = enc
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        with torch.no_grad():
            enc_out = self.enc(obs_geo)
        conv_out = self.conv(enc_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorSO2_1(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
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

class EquivariantSACActorSO2_2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_2
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted(),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
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

class EquivariantPolicySO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        return mean

# non-equi non-inv theta
class EquivariantSACActor2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.N = N
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act,
                                   1 * [self.c4_act.irrep(1)] +
                                   1 * [self.c4_act.regular_repr] +
                                   (action_dim*2-3) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3]
        inv_act = conv_out[:, 2+self.N:2+self.N+self.action_dim-3]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:], dtheta), dim=1)
        log_std = conv_out[:, -self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

# rho(1) std x y
class EquivariantSACActor3(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 2 * [self.c4_act.irrep(1)] + (action_dim*2-4) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        log_std_xy = conv_out[:, 2:4]
        inv_act = conv_out[:, 4:4+self.action_dim-2]
        inv_log_std = conv_out[:, 4+self.action_dim-2:]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = torch.cat((inv_log_std[:, 0:1], log_std_xy, inv_log_std[:, 1:]), dim=1)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantPolicy(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.N = N
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        return torch.tanh(mean)

class EquivariantSACVecCriticBase(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_rho1 = (obs_dim - 1) // 4
        self.num_obs_inv = obs_dim - 2*self.num_obs_rho1
        self.act = None
        self.q1 = None
        self.q2 = None

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_p = obs[:, 0:1]
        obs_rho1s = []
        obs_invs = []
        for i in range(self.num_obs_rho1):
            obs_rho1s.append(obs[:, 1+i*4:1+i*4+2])
            obs_invs.append(obs[:, 1+i*4+2:1+i*4+4])
        obs_rho1s = torch.cat(obs_rho1s, 1)
        obs_invs = torch.cat(obs_invs, 1)

        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)

        inp = torch.cat((dxy, obs_rho1s, inv_act, obs_p, obs_invs), dim=1).reshape(batch_size, -1, 1, 1)
        inp_geo = nn.GeometricTensor(inp, nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]))
        out1 = self.q1(inp_geo).tensor.reshape(batch_size, 1)
        out2 = self.q2(inp_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACVecCritic(EquivariantSACVecCriticBase):
    def __init__(self, obs_dim=7, action_dim=5, n_hidden=1024, N=4, initialize=True):
        super().__init__(obs_dim, action_dim)
        self.act = gspaces.Rot2dOnR2(N)
        self.q1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.act, n_hidden * [self.act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.trivial_repr]),
                      nn.FieldType(self.act, 1 * [self.act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
        self.q2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.act, n_hidden * [self.act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.trivial_repr]),
                      nn.FieldType(self.act, 1 * [self.act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

class EquivariantSACVecGaussianPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_dim=7, action_dim=5, n_hidden=1024, N=4, initialize=True):
        super().__init__()
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_rho1 = (obs_dim - 1) // 4
        self.num_obs_inv = obs_dim - 2*self.num_obs_rho1
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, self.num_obs_rho1 * [self.c4_act.irrep(1)] + self.num_obs_inv * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_p = obs[:, 0:1]
        obs_rho1s = []
        obs_invs = []
        for i in range(self.num_obs_rho1):
            obs_rho1s.append(obs[:, 1+i*4:1+i*4+2])
            obs_invs.append(obs[:, 1+i*4+2:1+i*4+4])
        obs_rho1s = torch.cat(obs_rho1s, 1)
        obs_invs = torch.cat(obs_invs, 1)
        inp = torch.cat((obs_rho1s, obs_p, obs_invs), dim=1).reshape(batch_size, -1, 1, 1)
        inp_geo = nn.GeometricTensor(inp, nn.FieldType(self.c4_act, self.num_obs_rho1 * [self.c4_act.irrep(1)] + self.num_obs_inv * [self.c4_act.trivial_repr]))
        conv_out = self.conv(inp_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    critic = EquivariantSACCriticSO2_2(obs_shape=(2, 128, 128), action_dim=5, n_hidden=32, initialize=False)
    o = torch.zeros(1, 2, 128, 128)
    o[0, 0, 10:20, 10:20] = 1
    a = torch.zeros(1, 5)
    a[0, 1:3] = torch.tensor([-1., -1.])

    o2 = torch.rot90(o, 1, [2, 3])
    a2 = torch.zeros(1, 5)
    a2[0, 1:3] = torch.tensor([1., -1.])

    out = critic(o, a)

    actor = EquivariantSACActorSO2_2(obs_shape=(2, 128, 128), action_dim=5, n_hidden=32, initialize=False)
    out2 = actor(o)
    print(1)
    # actor = EquivariantSACActor2(obs_shape=(2, 128, 128), action_dim=5, n_hidden=64, initialize=False)
    # out3 = actor(o)
    #
    # critic = EquivariantSACCritic(obs_shape=(2, 64, 64), action_dim=4, n_hidden=64, initialize=False)
    # o = torch.zeros(1, 2, 64, 64)
    # o[0, 0, 10:20, 10:20] = 1
    # a = torch.zeros(1, 4)
    # a[0, 1:3] = torch.tensor([-1., -1.])
    #
    # o2 = torch.rot90(o, 1, [2, 3])
    # a2 = torch.zeros(1, 4)
    # a2[0, 1:3] = torch.tensor([1., -1.])
    #
    # out = critic(o, a)
    #
    # actor = EquivariantSACActor(obs_shape=(2, 64, 64), action_dim=5, n_hidden=64, initialize=False)
    # out2 = actor(o)
    # actor = EquivariantSACActor2(obs_shape=(2, 64, 64), action_dim=5, n_hidden=64, initialize=False)
    # out3 = actor(o)

    critic = EquivariantSACVecCritic(obs_dim=5, action_dim=5, n_hidden=64, initialize=True)
    obs = torch.zeros(1, 5)
    obs[0, 1] = 1
    obs[0, 2] = 0
    act = torch.zeros(1, 5)
    act[0, 1] = 1
    act[0, 2] = 0
    out1 = critic(obs, act)

    obs = torch.zeros(1, 5)
    obs[0, 1] = 0
    obs[0, 2] = 1
    act = torch.zeros(1, 5)
    act[0, 1] = 0
    act[0, 2] = 1
    out2 = critic(obs, act)

    obs = torch.zeros(1, 5)
    obs[0, 1] = 1
    obs[0, 2] = 0
    act = torch.zeros(1, 5)
    act[0, 1] = 0
    act[0, 2] = 1
    out3 = critic(obs, act)

    actor = EquivariantSACVecGaussianPolicy(obs_dim=5, action_dim=5, n_hidden=64, initialize=False)
    out5 = actor(obs)
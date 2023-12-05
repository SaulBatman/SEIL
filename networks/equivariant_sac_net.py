import torch
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn

from networks.sac_networks import SACGaussianPolicyBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

from networks.cnn import SpatialSoftArgmax

class SpatialSoftArgmaxCycle(torch.nn.Module):
    """Spatial softmax as defined in https://arxiv.org/abs/1504.00702.

    Concretely, the spatial softmax of each feature map is used to compute a weighted
    mean of the pixel locations, effectively performing a soft arg-max over the feature
    dimension.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then apply the
        # softmax operator over the last dimension.
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then sum over
        # the h*w dimension. This effectively computes the weighted mean x and y
        # locations.
        mean = (softmax * torch.stack([torch.abs(xc), torch.abs(yc)]).max(0)[0].flatten()).sum(dim=1, keepdims=True)
        return mean.view(-1, c)
        # x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        # y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
        # x_mean_neg = (softmax * -xc.flatten()).sum(dim=1, keepdims=True)
        # y_mean_neg = (softmax * -yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        # return torch.cat([x_mean, y_mean, x_mean_neg, y_mean_neg], dim=1).view(-1, c * 4)

class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, initialize=True):
        super(EquiResBlock, self).__init__()
        r2_act = gspaces.rot2dOnR2(N=N)
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
        self.c4_act = gspaces.rot2dOnR2(N)
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
        self.d4_act = gspaces.flipRot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, obs_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128DihedralK5(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.d4_act = gspaces.flipRot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, obs_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 14x14
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]), inplace=True),
            # 12x12
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 10x10
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 5x5
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128SO2_1(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
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
        self.c4_act = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
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

class EquivariantEncoder128SO2_3(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.so2.irrep(0), self.so2.irrep(1), self.so2.irrep(2), self.so2.irrep(3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.so2, obs_channel * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.so2, n_out // 8 * self.repr),
                      nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.so2, n_out // 4 * self.repr),
                      nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.so2, n_out // 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.so2, n_out * 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2_3(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128NoPool(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.rot2dOnR2(N)
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
        self.c4_act = gspaces.rot2dOnR2(N)
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
        self.c4_act = gspaces.rot2dOnR2(N)
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
        self.c4_act = gspaces.rot2dOnR2(N)
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
        self.c4_act = gspaces.rot2dOnR2(N)
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






class EquivariantPolicy(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.N = N
        self.action_dim = action_dim
        self.c4_act = gspaces.rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        normalized_mean = torch.tanh(mean)
        return normalized_mean


class EquivariantPolicyDihedral(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.d4_act = gspaces.flipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))] + (action_dim * 2 - 3) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.d4_act, self.obs_channel * [self.d4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim + 1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        normalized_mean = torch.tanh(mean)
        return normalized_mean



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    ebm = EquivariantEBMDihedralSpatialSoftmax(obs_shape=(2, 128, 128), action_dim=5, n_hidden=8, N=4, initialize=True)
    o = torch.zeros(1, 2, 128, 128)
    o[0, 0, 10:20, 10:20] = 1
    a = torch.zeros(1, 1, 5)
    a[0, 0, 1:3] = torch.tensor([-1., 1.])

    o2 = torch.rot90(o, 1, [2, 3])
    a2 = torch.zeros(1, 1, 5)
    a2[0, 0, 1:3] = torch.tensor([-1., -1.])

    out = ebm(o, a)
    out2 = ebm(o2, a2)

    assert (out - out2) < 1e-4

    policy = EquivariantPolicyDihedralSpatialSoftmax2(obs_shape=(2, 128, 128), action_dim=5, n_hidden=16, N=4, initialize=True)
    policy(o)

    critic = EquivariantSACCriticO2_3(obs_shape=(2, 128, 128), action_dim=5, n_hidden=32, initialize=False)
    o = torch.zeros(1, 2, 128, 128)
    o[0, 0, 10:20, 10:20] = 1
    a = torch.zeros(1, 5)
    a[0, 1:3] = torch.tensor([-1., -1.])

    o2 = torch.rot90(o, 1, [2, 3])
    a2 = torch.zeros(1, 5)
    a2[0, 1:3] = torch.tensor([1., -1.])

    out = critic(o, a)

    actor = EquivariantSACActorO2_3(obs_shape=(2, 128, 128), action_dim=5, n_hidden=32, initialize=False)
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
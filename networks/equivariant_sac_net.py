import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn

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

class EquivariantEncoder128SO2_3(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
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
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
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
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
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
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
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
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-3) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)] + 1*[self.c4_act.quotient_repr((None, 4))]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-3) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)] + 1*[self.c4_act.quotient_repr((None, 4))]),
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
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]  + 1*[self.c4_act.quotient_repr((None, 4))]))
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

class EquivariantSACCriticSO2_3(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_3
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.so2.irrep(0), self.so2.irrep(1), self.so2.irrep(2), self.so2.irrep(3)]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * self.repr + (action_dim - 2) * [self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]),
                      nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, 1 * [self.so2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * self.repr + (action_dim - 2) * [self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]),
                      nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.so2, n_hidden*4 * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, 1 * [self.so2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.so2, self.obs_channel * [self.so2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.so2, self.n_hidden * self.repr + n_inv * [self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2_2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2_2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2_3(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2_3
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
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
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + 1 * [self.c4_act.quotient_repr((None, 4))] + (action_dim*2-3) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim+1:]
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

class EquivariantSACActorSO2_3(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_3
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * self.repr),
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

class EquivariantSACActorO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorO2_2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2_2
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorO2_3(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2_3
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
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

class EquivariantPolicyO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim - 2) * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
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
        return torch.tanh(mean)

<<<<<<< HEAD
class EquivariantPolicyDihedral(torch.nn.Module):
=======
class EquivariantPolicyDihedral(SACGaussianPolicyBase):
>>>>>>> 2786cca07681269677621d3c8d06544ce71c8581
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
<<<<<<< HEAD
        self.N = N
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
=======
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
>>>>>>> 2786cca07681269677621d3c8d06544ce71c8581
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
<<<<<<< HEAD
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1,1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
=======
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + 1 * [self.c4_act.quotient_repr((None, 4))] + (action_dim*2-3) * [self.c4_act.trivial_repr]),
>>>>>>> 2786cca07681269677621d3c8d06544ce71c8581
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
<<<<<<< HEAD
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        return torch.tanh(mean)
=======
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim + 1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        return mean

class EquivariantPolicyDihedralSpatialSoftmax1(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden
        self.img_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_shape[0] * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]), inplace=True)
        )
        self.reducer = SpatialSoftArgmax()
        self.out_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + 1 * [self.c4_act.quotient_repr((None, 4))] + (action_dim*2-3) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo).tensor
        conv_out = self.reducer(conv_out)
        conv_out = conv_out.reshape(batch_size, self.n_hidden * 2, 1, 1)
        conv_out = nn.GeometricTensor(conv_out, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.irrep(1, 1)]))
        conv_out = self.out_conv(conv_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        return mean

class EquivariantPolicyDihedralSpatialSoftmax(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden
        self.N = N
        self.img_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_shape[0] * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True)
        )
        self.reducer = SpatialSoftArgmaxCycle()
        self.out_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + 1 * [self.c4_act.quotient_repr((None, 4))] + (action_dim*2-3) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo).tensor
        conv_out = self.reducer(conv_out)
        conv_out = conv_out.reshape(batch_size, self.n_hidden * self.c4_act.regular_repr.size, 1, 1)
        conv_out = nn.GeometricTensor(conv_out, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr]))
        conv_out = self.out_conv(conv_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim + 1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        return mean
>>>>>>> 2786cca07681269677621d3c8d06544ce71c8581

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

class EquivariantEBMDihedral(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.cat_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-3) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)] + 1*[self.c4_act.quotient_repr((None, 4))]),
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
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))

        fused = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1), dxy.reshape(batch_size, act.size(1), 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused.size()
        fused = fused.reshape(B * N, D, 1, 1)
        fused_geo = nn.GeometricTensor(fused, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))

        # cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        # cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        out = self.cat_conv(fused_geo).tensor.reshape(B, N)
        return out

# from networks.cnn import SpatialSoftArgmaxCycle
# class EquivariantEBMDihedralSpatialSoftmax(torch.nn.Module):
#     def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
#         super().__init__()
#         assert kernel_size in [3, 5]
#         self.obs_channel = obs_shape[0]
#         self.n_hidden = n_hidden
#         self.c4_act = gspaces.FlipRot2dOnR2(N)
#         self.img_conv = torch.nn.Sequential(
#             # 128x128
#             nn.R2Conv(nn.FieldType(self.c4_act, obs_shape[0] * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), 2),
#             # 64x64
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), 2),
#             # 32x32
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), 2),
#             # 16x16
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), 2),
#             # 8x8
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]), inplace=True)
#         )
#
#         self.reducer = SpatialSoftArgmaxCycle()
#
#         self.n_rho1 = 2 if N==2 else 1
#
#         self.cat_conv = torch.nn.Sequential(
#             nn.R2Conv(nn.FieldType(self.c4_act, (n_hidden + self.n_rho1) * [self.c4_act.irrep(1, 1)] + (action_dim-2) * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
#             nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#         )
#
#     def forward(self, obs, act):
#         batch_size = obs.shape[0]
#         obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
#         conv_out = self.img_conv(obs_geo).tensor
#         conv_out = self.reducer(conv_out)
#         conv_out = conv_out.reshape(batch_size, self.n_hidden * 2, 1, 1)
#         dxy = act[:, :, 1:3]
#         inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:]), dim=2)
#         n_inv = inv_act.shape[2]
#
#         fused = torch.cat([conv_out.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), dxy.reshape(batch_size, act.size(1), 2, 1, 1), inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1)], dim=2)
#         B, N, D, _, _ = fused.size()
#         fused = fused.reshape(B * N, D, 1, 1)
#         fused_geo = nn.GeometricTensor(fused, nn.FieldType(self.c4_act, (self.n_hidden + self.n_rho1) * [self.c4_act.irrep(1, 1)] + n_inv * [self.c4_act.trivial_repr]))
#
#         out = self.cat_conv(fused_geo).tensor.reshape(B, N)
#         return out

class EquivariantEBMDihedralSpatialSoftmax(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        self.img_conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_shape[0] * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden // 2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), 8),
        )

        self.reducer = SpatialSoftArgmaxCycle()

        self.n_rho1 = 2 if N==2 else 1

        self.cat_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-3) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)] + 1*[self.c4_act.quotient_repr((None, 4))]),
            # nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
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
        conv_out = self.img_conv(obs_geo).tensor
        conv_out = self.reducer(conv_out)
        conv_out = conv_out.reshape(batch_size, self.n_hidden * self.c4_act.regular_repr.size, 1, 1)
        # conv_out = conv_out.permute(0, 2, 1, 3, 4)
        # conv_out = conv_out.reshape(batch_size, 2 * self.n_hidden * self.N * 2, 1, 1)
        dxy = act[:, :, 1:3]
        inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:4]), dim=2)
        dtheta = act[:, :, 4:5]
        n_inv = inv_act.shape[2]

        fused = torch.cat([conv_out.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1),
                           inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1),
                           dxy.reshape(batch_size, act.size(1), 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1, 1),
                           (-dtheta).reshape(batch_size, 1, 1, 1, 1)], dim=2)

        B, N, D, _, _ = fused.size()
        fused = fused.reshape(B * N, D, 1, 1)
        fused_geo = nn.GeometricTensor(fused, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)] + 1*[self.c4_act.quotient_repr((None, 4))]))
        # fused_geo = nn.GeometricTensor(fused, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))

        out = self.cat_conv(fused_geo).tensor.reshape(B, N)
        return out

class EquivariantEBMDihedralFac(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.cat_conv_equ = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.cat_conv_inv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim - 2) * [self.c4_act.trivial_repr]),
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
        dxy = act[:, :, 1:3]
        inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:]), dim=2)
        n_inv = inv_act.shape[2]

        fused_equ = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), dxy.reshape(batch_size, act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        equ_out = self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr]))
        inv_out = self.cat_conv_inv(fused_inv_geo).tensor.reshape(B, N)
        return equ_out, inv_out

    def forwardEqu(self, obs, equ_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)

        fused_equ = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, equ_act.size(1), -1, -1, -1),
                               equ_act.reshape(batch_size, equ_act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]))
        return self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

    def forwardInv(self, obs, inv_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, inv_act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, inv_act.size(1), 3, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + 3 * [self.c4_act.trivial_repr]))
        return self.cat_conv_inv(fused_inv_geo).tensor.reshape(B, N)

class EquivariantEBMDihedralFacAll(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.cat_conv_equ = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.cat_conv_inv_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.cat_conv_inv_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.cat_conv_inv_3 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]),
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
        dxy = act[:, :, 1:3]
        inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:]), dim=2)

        fused_equ = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), dxy.reshape(batch_size, act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        equ_out = self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

        fused_inv_1 = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act[:, :, 0].reshape(batch_size, act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv_1.size()
        fused_inv_1 = fused_inv_1.reshape(B * N, D, 1, 1)
        fused_inv_geo_1 = nn.GeometricTensor(fused_inv_1, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        inv_out_1 = self.cat_conv_inv_1(fused_inv_geo_1).tensor.reshape(B, N)

        fused_inv_2 = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act[:, :, 1].reshape(batch_size, act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv_2.size()
        fused_inv_2 = fused_inv_2.reshape(B * N, D, 1, 1)
        fused_inv_geo_2 = nn.GeometricTensor(fused_inv_2, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        inv_out_2 = self.cat_conv_inv_2(fused_inv_geo_2).tensor.reshape(B, N)

        fused_inv_3 = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act[:, :, 2].reshape(batch_size, act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv_3.size()
        fused_inv_3 = fused_inv_3.reshape(B * N, D, 1, 1)
        fused_inv_geo_3 = nn.GeometricTensor(fused_inv_3, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        inv_out_3 = self.cat_conv_inv_3(fused_inv_geo_3).tensor.reshape(B, N)
        return equ_out, inv_out_1, inv_out_2, inv_out_3

    def forwardEqu(self, obs, equ_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)

        fused_equ = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, equ_act.size(1), -1, -1, -1),
                               equ_act.reshape(batch_size, equ_act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]))
        return self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

    def forwardInv1(self, obs, inv_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, inv_act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, inv_act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        return self.cat_conv_inv_1(fused_inv_geo).tensor.reshape(B, N)

    def forwardInv2(self, obs, inv_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, inv_act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, inv_act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        return self.cat_conv_inv_2(fused_inv_geo).tensor.reshape(B, N)

    def forwardInv3(self, obs, inv_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, inv_act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, inv_act.size(1), 1, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + 1 * [self.c4_act.trivial_repr]))
        return self.cat_conv_inv_3(fused_inv_geo).tensor.reshape(B, N)

class EquivariantEBMDihedralFacSepEnc(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.FlipRot2dOnR2(N)
        enc = EquivariantEncoder128Dihedral if kernel_size == 3 else EquivariantEncoder128DihedralK5
        self.img_conv_equ = enc(self.obs_channel, n_hidden, initialize, N)
        self.img_conv_inv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.cat_conv_equ = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.cat_conv_inv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim - 2) * [self.c4_act.trivial_repr]),
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
        conv_equ = self.img_conv_equ(obs_geo)
        conv_inv = self.img_conv_inv(obs_geo)
        dxy = act[:, :, 1:3]
        inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:]), dim=2)
        n_inv = inv_act.shape[2]

        fused_equ = torch.cat([conv_equ.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), dxy.reshape(batch_size, act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + self.n_rho1*[self.c4_act.irrep(1, 1)]))
        equ_out = self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

        fused_inv = torch.cat([conv_inv.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1), inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr]))
        inv_out = self.cat_conv_inv(fused_inv_geo).tensor.reshape(B, N)
        return equ_out, inv_out

    def forwardEqu(self, obs, equ_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.img_conv_equ(obs_geo)

        fused_equ = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, equ_act.size(1), -1, -1, -1),
                               equ_act.reshape(batch_size, equ_act.size(1), 2, 1, 1)], dim=2)
        B, N, D, _, _ = fused_equ.size()
        fused_equ = fused_equ.reshape(B * N, D, 1, 1)
        fused_equ_geo = nn.GeometricTensor(fused_equ, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]))
        return self.cat_conv_equ(fused_equ_geo).tensor.reshape(B, N)

    def forwardInv(self, obs, inv_act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv_inv(obs_geo)
        fused_inv = torch.cat([conv_out.tensor.unsqueeze(1).expand(-1, inv_act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, inv_act.size(1), 3, 1, 1)], dim=2)
        B, N, D, _, _ = fused_inv.size()
        fused_inv = fused_inv.reshape(B * N, D, 1, 1)
        fused_inv_geo = nn.GeometricTensor(fused_inv, nn.FieldType(self.c4_act, self.n_hidden * [
            self.c4_act.regular_repr] + 3 * [self.c4_act.trivial_repr]))
        return self.cat_conv_inv(fused_inv_geo).tensor.reshape(B, N)

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

    policy = EquivariantPolicyDihedralSpatialSoftmax(obs_shape=(2, 128, 128), action_dim=5, n_hidden=8, N=4, initialize=True)
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
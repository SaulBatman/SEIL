import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFac(nn.Module):
    def __init__(self, n_input_channel=2, n_p=2, n_theta=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(n_input_channel, 32, kernel_size=3, padding=1),
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
            torch.nn.Linear(512*8*8, 1024),
            nn.ReLU(inplace=True),
        )

        self.dp_fc = torch.nn.Linear(1024, n_p)
        self.dxy_fc = torch.nn.Linear(1024, 9)
        self.dz_fc = torch.nn.Linear(1024, 3)
        self.dtheta_fc = torch.nn.Linear(1024, n_theta)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        h = self.conv(x)
        dp = self.dp_fc(h)
        dxy = self.dxy_fc(h)
        dz = self.dz_fc(h)
        dtheta = self.dtheta_fc(h)
        return dp, dxy, dz, dtheta

class CNNCom(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        conv_out_size = ((((input_shape[1] // 2) // 2) // 2) // 2)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
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
            torch.nn.Linear(512 * conv_out_size * conv_out_size, 1024),
            nn.ReLU(inplace=True),
        )

        self.n_p = n_p
        self.n_theta = n_theta
        self.fc = torch.nn.Linear(1024, 9 * 3 * n_theta * n_p)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        h = self.conv(x)
        q = self.fc(h)
        return q

# similar amount of parameters
class CNNCom2(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
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
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(512, 18, kernel_size=1, padding=0),
        )

        self.n_p = n_p
        self.n_theta = n_theta

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.conv(x)
        q = q.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return q

class DQNComCURL(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1, curl_z=128):
        super().__init__()
        conv_out_size = ((((input_shape[1]//2)//2)//2)//2)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
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
            torch.nn.Linear(512 * conv_out_size * conv_out_size, 1024),
            nn.ReLU(inplace=True),
        )

        self.W_h = nn.Parameter(torch.rand(1024, 256))
        self.layer_norm_1 = nn.LayerNorm(256)
        self.W_c = nn.Parameter(torch.rand(256, 128))
        self.b_h = nn.Parameter(torch.zeros(256))
        self.b_c = nn.Parameter(torch.zeros(128))
        self.W = nn.Parameter(torch.rand(128, 128))
        self.layer_norm_2 = nn.LayerNorm(128)

        self.n_p = n_p
        self.n_theta = n_theta
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 9 * 3 * n_theta * n_p),
        )

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        q = self.fc(x)

        h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
        h = self.layer_norm_1(h)
        h = F.relu(h)
        h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
        h = self.layer_norm_2(h)
        return q, h

class DQNComCURLOri(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        x = torch.randn([1] + list(input_shape))
        conv_out_dim = self.conv(x).reshape(-1).shape[-1]

        self.W_h = nn.Parameter(torch.rand(conv_out_dim, 256))
        self.layer_norm_1 = nn.LayerNorm(256)
        self.W_c = nn.Parameter(torch.rand(256, 128))
        self.b_h = nn.Parameter(torch.zeros(256))
        self.b_c = nn.Parameter(torch.zeros(128))
        self.W = nn.Parameter(torch.rand(128, 128))
        self.layer_norm_2 = nn.LayerNorm(128)

        self.n_p = n_p
        self.n_theta = n_theta
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            torch.nn.Linear(256, 9 * 3 * n_theta * n_p),
        )

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        q = self.fc(x)

        h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
        h = self.layer_norm_1(h)
        h = F.relu(h)
        h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
        h = self.layer_norm_2(h)
        return q, h

# class DQNComCURL(nn.Module):
#     def __init__(self, n_p=2, n_theta=1, curl_z=128):
#         super().__init__()
#         self.conv = torch.nn.Sequential(
#             nn.Conv2d(2, 32, 5, stride=5, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 5, stride=5, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         self.W_h = nn.Parameter(torch.rand(1600, 256))
#         self.layer_norm_1 = nn.LayerNorm(256)
#         self.W_c = nn.Parameter(torch.rand(256, 128))
#         self.b_h = nn.Parameter(torch.zeros(256))
#         self.b_c = nn.Parameter(torch.zeros(128))
#         self.W = nn.Parameter(torch.rand(128, 128))
#         self.layer_norm_2 = nn.LayerNorm(128)
#
#         self.n_p = n_p
#         self.n_theta = n_theta
#         self.fc = torch.nn.Sequential(
#             nn.Linear(1600, 256),
#             nn.ReLU(),
#             nn.Linear(256, 9 * 3 * n_theta * n_p),
#         )
#
#         for m in self.named_modules():
#             if isinstance(m[1], nn.Conv2d):
#                 # nn.init.kaiming_normal_(m[1].weight.data)
#                 nn.init.xavier_normal_(m[1].weight.data)
#             elif isinstance(m[1], nn.BatchNorm2d):
#                 m[1].weight.data.fill_(1)
#                 m[1].bias.data.zero_()
#
#     def forward(self, x):
#         x = self.conv(x)
#         q = self.fc(x)
#
#         h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
#         # h = F.layer_norm(h, list(h.shape))
#         h = self.layer_norm_1(h)
#         h = F.relu(h)
#         h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
#         h = self.layer_norm_2(h)
#         return q, h

class Actor(nn.Module):
    def __init__(self, action_dim=5):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
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

            # nn.Flatten(),
            # torch.nn.Linear(512*8*8, 1024),
            # nn.ReLU(inplace=True),
            SpatialSoftArgmax(),
            torch.nn.Linear(1024, action_dim),
            nn.Tanh()
        )

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        return self.conv(x)

# class Critic(nn.Module):
#     def __init__(self, action_dim=5):
#         super().__init__()
#         self.state_conv = torch.nn.Sequential(
#             # 128x128
#             nn.Conv2d(2, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             # 64x64
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             # 32x32
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             # 16x16
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             # 8x8
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(),
#             torch.nn.Linear(512 * 8 * 8, 1024),
#             nn.ReLU(inplace=True),
#         )
#
#         self.action_fc = torch.nn.Sequential(
#             torch.nn.Linear(action_dim, 1024),
#             nn.ReLU(inplace=True),
#         )
#
#         self.critic_fc = torch.nn.Sequential(
#             torch.nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             torch.nn.Linear(1024, 1)
#         )
#
#     def forward(self, obs, act):
#         obs_out = self.state_conv(obs)
#         act_out = self.action_fc(act)
#         out = self.critic_fc(torch.cat((obs_out, act_out), dim=1))
#         return out

class Critic(nn.Module):
    def __init__(self, action_dim=5):
        super().__init__()
        self.state_conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
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
            torch.nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
        )

        self.critic_fc = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

    def forward(self, obs, act):
        obs_out = self.state_conv(obs)
        out = self.critic_fc(torch.cat((obs_out, act), dim=1))
        return out

class SpatialSoftArgmax(nn.Module):
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
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class CNNEBM(nn.Module):
    def __init__(self, action_dim=5):
        super().__init__()
        self.state_conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
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
        )

        self.reducer = SpatialSoftArgmax()
        # self.reducer = torch.nn.Sequential(
        #     nn.Flatten(),
        #     torch.nn.Linear(512*8*8, 1024),
        #     nn.ReLU(inplace=True),
        # )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

    def forward(self, obs, act):
        out = self.state_conv(obs)
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, act.size(1), -1), act], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.fc(fused)
        return out.view(B, N)

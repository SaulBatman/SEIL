import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, action_dim=5, reducer='maxpool'):
        assert reducer in ['maxpool', 'spatial_softmax']
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
        )

        if reducer == 'maxpool':
            self.reducer = torch.nn.Sequential(
                nn.MaxPool2d(8),
                nn.Flatten()
            )
            mlp_in = 256
        elif reducer == 'spatial_softmax':
            self.reducer = SpatialSoftArgmax()
            mlp_in = 512
        else:
            raise NotImplementedError

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(mlp_in+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1),
        )

    def forward(self, obs, act):
        out = self.state_conv(obs)
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, act.size(1), -1), act], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.fc(fused)
        return out.view(B, N)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sm = SpatialSoftArgmax()
    inp = torch.zeros(1, 1, 100, 100)
    inp[:, :, 10:30, 10:20] = 1
    plt.imshow(inp[0, 0])
    plt.show()
    out = sm(inp)
    print(out)

    inp = torch.zeros(1, 1, 100, 100)
    inp[:, :, -30:-10, 10:20] = 1
    plt.imshow(inp[0, 0])
    plt.show()
    out = sm(inp)
    print(out)

    pass
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet"]


def _num_groups(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):

        if channels % groups == 0:
            return groups

    return 1


def _make_norm(norm, channels):
    if norm == "group":
        return nn.GroupNorm(_num_groups(channels), channels)

    if norm == "batch":
        return nn.BatchNorm2d(channels)

    raise ValueError(f"Unknown norm {norm!r}; expected 'group' or 'batch'.")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm="group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm="group"):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, norm)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm="group"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, norm)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, norm="group", head_bias=True):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c, norm)
        self.down1 = Down(c, c * 2, norm)
        self.down2 = Down(c * 2, c * 4, norm)
        self.down3 = Down(c * 4, c * 8, norm)
        self.down4 = Down(c * 8, c * 16, norm)
        self.up1 = Up(c * 16, c * 8, norm)
        self.up2 = Up(c * 8, c * 4, norm)
        self.up3 = Up(c * 4, c * 2, norm)
        self.up4 = Up(c * 2, c, norm)
        self.head = nn.Conv2d(c, n_classes, kernel_size=1, bias=head_bias)
        self._initialize_weights()

    def forward_features(self, x, return_bottleneck=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        z = self.up4(d, x1)

        if return_bottleneck:
            return z, x5

        return z

    def forward_head(self, z):
        return self.head(z)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

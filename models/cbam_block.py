import torch
import torch.nn as nn


class ImprovedChannelAttention2D(nn.Module):
    def __init__(self, in_planes):
        super(ImprovedChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        kernel_size = max(3, min(9, in_planes // 4))
        if kernel_size % 2 == 0:
            kernel_size += 1

        padding = (kernel_size - 1) // 2


        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.avg_pool(x).view(b, 1, c)
        max_out = self.max_pool(x).view(b, 1, c)

        avg_out = self.conv1d(avg_out)
        max_out = self.conv1d(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2D, self).__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.size(2) < self.kernel_size or x.size(3) < self.kernel_size:
            if not hasattr(self, 'small_conv'):
                small_kernel = 3
                small_padding = 1
                self.small_conv = nn.Conv2d(2, 1, kernel_size=small_kernel,
                                            padding=small_padding, bias=False).to(x.device)
                nn.init.kaiming_normal_(self.small_conv.weight)

            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            concat = torch.cat([avg_out, max_out], dim=1)
            attn = self.sigmoid(self.small_conv(concat))
            return x * attn

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class ImprovedCBAM2D(nn.Module):
    def __init__(self, channels):
        super(ImprovedCBAM2D, self).__init__()
        self.ca = ImprovedChannelAttention2D(channels)
        self.sa = SpatialAttention2D(kernel_size=3)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
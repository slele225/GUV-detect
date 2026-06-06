"""Small U-Net CenterNet-style model for GUV detection.

A compact U-Net backbone with two 1x1-conv heads:
  - center heatmap (1 channel, sigmoid -> probability of a GUV center)
  - radius        (1 channel, raw -> predicted radius in input pixels)

The output is at stride `out_stride` (default 1 = full resolution) so close
centers in crowded fields stay separable. Deliberately small -- easily trains on
a single H100, and runs on CPU for tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GUVNet(nn.Module):
    """U-Net with heatmap + radius heads.

    Args:
        in_ch: input channels (1, lipid).
        base: channel width of the first level (doubles each downsample).
        depth: number of downsampling levels.
        out_stride: output stride (1 = full res). Must be a power of two and
            <= 2**depth; the decoder stops upsampling once it reaches it.
    """

    def __init__(self, in_ch: int = 1, base: int = 32, depth: int = 4, out_stride: int = 1):
        super().__init__()
        assert out_stride >= 1 and (out_stride & (out_stride - 1)) == 0, "out_stride must be a power of 2"
        self.depth = depth
        self.out_stride = out_stride
        # log2(out_stride) for a power of two = bit_length - 1; stop the decoder
        # that many levels early so the output sits at the requested stride.
        self.n_up = depth - (out_stride.bit_length() - 1)
        assert self.n_up >= 0, "out_stride too large for this depth"

        chans = [base * (2**i) for i in range(depth + 1)]  # e.g. [32,64,128,256,512]

        self.inc = _DoubleConv(in_ch, chans[0])
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), _DoubleConv(chans[i], chans[i + 1])) for i in range(depth)]
        )
        # Decoder: upsample + concat skip + double conv, for n_up levels.
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        cur = chans[depth]
        for j in range(self.n_up):
            skip_ch = chans[depth - 1 - j]
            self.ups.append(nn.Conv2d(cur, skip_ch, 1))  # channel reduce before concat
            self.up_convs.append(_DoubleConv(skip_ch * 2, skip_ch))
            cur = skip_ch

        self.hm_head = nn.Conv2d(cur, 1, 1)
        self.radius_head = nn.Conv2d(cur, 1, 1)
        # Bias the heatmap toward background (rare positives) -- standard for focal.
        nn.init.constant_(self.hm_head.bias, -4.6)

    def forward(self, x) -> dict:
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))

        h = feats[-1]
        for j in range(self.n_up):
            skip = feats[self.depth - 1 - j]
            h = self.ups[j](F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
            h = self.up_convs[j](torch.cat([h, skip], dim=1))

        hm = torch.sigmoid(self.hm_head(h))
        hm = torch.clamp(hm, 1e-4, 1 - 1e-4)  # stabilize focal-loss logs
        radius = self.radius_head(h)
        return {"hm": hm, "radius": radius}

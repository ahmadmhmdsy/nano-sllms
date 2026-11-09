"""Compact DBNet-style PAW detector (Stage 2).

The output is at stride 2, not DBNet's usual stride 4: a PAW is a sub-word ink
component, and on the 200-page gate the median PAW is 15x24 px (min-side p25 =
7 px). At stride 4 that median box is under 6 px wide and the p25 box is under
2 px, which no detector can separate; at stride 2 the median box is 7x12 px.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + idt, inplace=True)


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PAWDetector(nn.Module):
    """FPN over a 4-stage residual backbone; two heads (prob, thresh) at stride 2."""

    def __init__(self, in_ch=1, widths=(24, 48, 96, 160), blocks=(1, 1, 2, 1), fpn=64,
                 out_stride=2, head_type="craft"):
        super().__init__()
        if int(out_stride) not in (1, 2):
            raise ValueError("out_stride must be 1 or 2, got %r" % (out_stride,))
        self.out_stride = int(out_stride)
        self.head_type = str(head_type)
        if self.head_type not in ("craft", "centernet"):
            raise ValueError("head_type must be 'craft' or 'centernet', got %r" % (head_type,))
        w1, w2, w3, w4 = widths
        self.stem = nn.Sequential(conv_bn(in_ch, w1, 3, 2), conv_bn(w1, w1, 3, 1))  # /2
        self.s1 = self._stage(w1, w2, blocks[0], 2)  # /4
        self.s2 = self._stage(w2, w3, blocks[1], 2)  # /8
        self.s3 = self._stage(w3, w4, blocks[2], 2)  # /16
        self.s4 = self._stage(w4, w4 * 2, blocks[3], 2)  # /32
        c4 = w4 * 2
        self.lat2 = nn.Conv2d(w2, fpn, 1)
        self.lat3 = nn.Conv2d(w3, fpn, 1)
        self.lat4 = nn.Conv2d(w4, fpn, 1)
        self.lat5 = nn.Conv2d(c4, fpn, 1)
        self.smooth = conv_bn(fpn, fpn, 3, 1)
        self.head = conv_bn(fpn, fpn, 3, 1)
        if self.head_type == "centernet":
            # One peak per PAW: the heatmap says where, the w/h head says how big.
            # No level set, so the box cannot inherit the blob's shape error and
            # adjacent PAWs cannot merge into a single component.
            self.heat = nn.Conv2d(fpn, 1, 1)
            self.size = nn.Conv2d(fpn, 2, 1)
            self.off = nn.Conv2d(fpn, 2, 1)
        else:
            self.prob = nn.Conv2d(fpn, 1, 1)
            self.thresh = nn.Conv2d(fpn, 1, 1)
        if self.out_stride == 1:
            # A PAW is 15-24 px and its CRAFT Gaussian has sigma = k*size ~= 1.7 px
            # at stride 2 -- sub-pixel for the network. A stride-1 level doubles it,
            # and the skip straight from the image restores the high-frequency
            # detail the strided backbone has already discarded.
            self.skip1 = conv_bn(in_ch, 16, 3, 1)
            self.head1 = conv_bn(fpn + 16, fpn, 3, 1)

    @staticmethod
    def _stage(cin, cout, n, stride):
        layers = [ResBlock(cin, cout, stride)]
        for _ in range(n - 1):
            layers.append(ResBlock(cout, cout, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.s1(c1)
        c3 = self.s2(c2)
        c4 = self.s3(c3)
        c5 = self.s4(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        f = self.smooth(p2)
        f = F.interpolate(f, scale_factor=2, mode="nearest")  # /2
        f = self.head(f)
        if self.out_stride == 1:
            f = F.interpolate(f, scale_factor=2, mode="nearest")  # /1
            f = self.head1(torch.cat([f, self.skip1(x)], dim=1))
        if self.head_type == "centernet":
            return {"heat": self.heat(f), "size": self.size(f), "off": self.off(f)}
        return {"prob": self.prob(f), "thresh": self.thresh(f)}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

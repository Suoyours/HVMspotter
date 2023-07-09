import os
import sys

import torch
from torch import nn


class SENet(nn.Module):
    def __init__(self, channel, ratio=4):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, c])  # b,c,h,w->b,c
        fc = self.fc(avg).view([b, c, 1, 1])  # b,c->b,c//ratio->b,c->b,c,1,1
        return x * fc


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_fc = self.fc(max_pool)
        avg_fc = self.fc(avg_pool)

        out = max_fc + avg_fc
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class SCEM_v1(nn.Module):
    def __init__(self, channel0, ratio=4):
        super(SCEM_v1, self).__init__()
        self.channel = channel0 * 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // ratio, False),
            nn.ReLU(),
            nn.Linear(self.channel // ratio, self.channel, False),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, f3, f4):
        b, c, h, w = f1.size()
        avg1 = self.avg_pool(f1).view([b, c])  # b,c,h,w->b,c
        avg2 = self.avg_pool(f2).view([b, c])  # b,c,h,w->b,c
        avg3 = self.avg_pool(f3).view([b, c])  # b,c,h,w->b,c
        avg4 = self.avg_pool(f4).view([b, c])  # b,c,h,w->b,c
        avg = torch.cat((avg1, avg2, avg3, avg4), dim=1)
        fc = self.fc(avg).view([b, c*4, 1, 1])  # b,c->b,c//ratio->b,c->b,c,1,1
        return fc


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

#
# x1 = torch.rand((1, 128, 160, 160))
# x2 = torch.rand((1, 128, 80, 80))
# x3 = torch.rand((1, 128, 40, 40))
# x4 = torch.rand((1, 128, 20, 20))
model = SCEM_v1(128)
# outputs = model(x1, x2, x3, x4)
# print(outputs)
print('Norm Conv parameter count is {}'.format(count_param(model)))
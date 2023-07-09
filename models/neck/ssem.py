import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..utils import Conv_BN_ReLU
from models.utils import Conv_BN_ReLU


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat((max_pool, avg_pool), dim=1)
        out = self.conv(pool_out)
        # out = self.sigmoid(out)
        # return out * x
        return out


class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_,
                                                                     f1_)))
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_,
                                                                     f2_)))
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4


class SSEM_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSEM_v1, self).__init__()
        planes = out_channels
        # attention usages
        self.sa_model1 = SpatialAttention()
        self.sa_model2 = SpatialAttention()
        self.sa_model3 = SpatialAttention()
        self.sa_model4 = SpatialAttention()
        self.conv0 = nn.Conv2d(4, 2, 3, 1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 3, 2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 3, 4, padding=1, bias=False)
        self.conv4 = nn.Conv2d(2, 1, 3, 8, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.dwconv3_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

    def _get_sa_weights(self, f1, f2, f3, f4):
        sa_weight1 = self.sa_model1(f1)
        sa_weight2 = self.sa_model2(f2)
        sa_weight3 = self.sa_model3(f3)
        sa_weight4 = self.sa_model4(f4)

        _, _, H, W = f1.size()
        sa_weight_cat = torch.cat((sa_weight1,
                                   F.interpolate(sa_weight2, size=(H, W), mode='bilinear'),
                                   F.interpolate(sa_weight3, size=(H, W), mode='bilinear'),
                                   F.interpolate(sa_weight4, size=(H, W), mode='bilinear')), dim=1)
        sa_weight_cat = self.conv0(sa_weight_cat)
        sa_weight1_ = self.conv1(sa_weight_cat)
        sa_weight2_ = self.conv2(sa_weight_cat)
        sa_weight3_ = self.conv3(sa_weight_cat)
        sa_weight4_ = self.conv4(sa_weight_cat)

        sa_weight1_ = self.sigmoid(sa_weight1_)
        sa_weight2_ = self.sigmoid(sa_weight2_)
        sa_weight3_ = self.sigmoid(sa_weight3_)
        sa_weight4_ = self.sigmoid(sa_weight4_)

        return sa_weight1_, sa_weight2_, sa_weight3_, sa_weight4_

    def _upsample_add(self, x, y, sa_w_x):
        _, _, H, W = y.size()
        return F.interpolate(x * sa_w_x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        sa_w1, sa_w2, sa_w3, sa_w4 = self._get_sa_weights(f1, f2, f3, f4)

        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3, sa_w4)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2, sa_w3)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1, sa_w2)))

        f1 = f1 + f1_ * sa_w1
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4
        return f1, f2, f3, f4


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


# x1 = torch.rand((1, 128, 160, 160))
# x2 = torch.rand((1, 128, 80, 80))
# x3 = torch.rand((1, 128, 40, 40))
# x4 = torch.rand((1, 128, 20, 20))
in_channels = (64, 128, 256, 512),
out_channels = 128
ssem = SSEM_v1(in_channels, out_channels)
# outputs1 = ssem(x1, x2, x3, x4)
# print(outputs1)
print('Norm Conv parameter count is {}'.format(count_param(ssem)))
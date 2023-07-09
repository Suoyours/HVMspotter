import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1  # 是否可以调用

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 残差结构实虚判断
            identity = self.downsample(x)  # 虚函数

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构
                 blocks_num,  # 残差结构数目
                 num_classes=1000,  # 训练集分类数目
                 include_top=True,  # 复杂可选
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 卷积核个数

        self.groups = groups  # 不用
        self.width_per_group = width_per_group  # 不用

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,  # 输入通道数 卷积核个数
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)  # 通道个数和卷核个数一样
        self.relu = nn.ReLU(inplace=True)  # 激活
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:  # 默认为Ture
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)#平均池化
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层 512*1 分类数

        for m in self.modules():  # 卷积层初始化！！！
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # 通过上面定义好的残差结构造层 残差结构 卷积核个数 残差结构个数 步长默认为一
        downsample = None  # 默认实
        if stride != 1 or self.in_channel != channel * block.expansion:  # 用不到
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,  # 第一层虚线残差结构
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):  # 后面全是实线残差结构
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)  # *列表去括号

    def forward(self, x):  # 正向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # 压平 变成一维矩阵
            x = self.fc(x)  # 全连接

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)



if __name__ == '__main__':
    import torch

    x = torch.zeros(2, 3, 640, 640)
    net = resnet18()
    y = net(x)

    for u in y:
        print(u.shape)

    print(net.out_channels)
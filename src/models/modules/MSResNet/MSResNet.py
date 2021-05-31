import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from . import pblm


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock5x5_1(nn.Module):
    expansion = 1

    def __init__(self, inplanes5_1, planes, stride=1, downsample=None):
        super(BasicBlock5x5_1, self).__init__()
        self.conv1 = conv5x5(inplanes5_1, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock5x5_2(nn.Module):
    expansion = 1

    def __init__(self, inplanes5_2, planes, stride=1, downsample=None):
        super(BasicBlock5x5_2, self).__init__()
        self.conv1 = conv5x5(inplanes5_2, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock5x5_3(nn.Module):
    expansion = 1

    def __init__(self, inplanes5_3, planes, stride=1, downsample=None):
        super(BasicBlock5x5_3, self).__init__()
        self.conv1 = conv5x5(inplanes5_3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class MSResNet(pblm.PrebuiltLightningModule):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        super().__init__(self.__class__.__name__)
        self.inplanes5_1 = 64
        self.inplanes5_2 = 64
        self.inplanes5_3 = 64

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer5x5_11 = self._make_layer5_1(BasicBlock5x5_1, 64, layers[0], stride=2)
        self.layer5x5_12 = self._make_layer5_1(BasicBlock5x5_1, 128, layers[1], stride=2)
        self.layer5x5_13 = self._make_layer5_1(BasicBlock5x5_1, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)
        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_1 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer5x5_21 = self._make_layer5_2(BasicBlock5x5_2, 64, layers[0], stride=2)
        self.layer5x5_22 = self._make_layer5_2(BasicBlock5x5_2, 128, layers[1], stride=2)
        self.layer5x5_23 = self._make_layer5_2(BasicBlock5x5_2, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_2 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer5x5_31 = self._make_layer5_3(BasicBlock5x5_3, 64, layers[0], stride=2)
        self.layer5x5_32 = self._make_layer5_3(BasicBlock5x5_3, 128, layers[1], stride=2)
        self.layer5x5_33 = self._make_layer5_3(BasicBlock5x5_3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_3 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(48384, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5_1(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_1, planes, stride, downsample))
        self.inplanes5_1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_1, planes))

        return nn.Sequential(*layers)

    def _make_layer5_2(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_2, planes, stride, downsample))
        self.inplanes5_2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_2, planes))

        return nn.Sequential(*layers)

    def _make_layer5_3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_3, planes, stride, downsample))
        self.inplanes5_3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_3, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer5x5_11(x0)
        x = self.layer5x5_12(x)
        x = self.layer5x5_13(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool5_1(x)

        y = self.layer5x5_21(x0)
        y = self.layer5x5_22(y)
        y = self.layer5x5_23(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5_2(y)

        z = self.layer5x5_31(x0)
        z = self.layer5x5_32(z)
        z = self.layer5x5_33(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool5_3(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.reshape(out.shape[0], -1)
        # out = self.drop(out)
        out1 = self.fc(out)

        return out1


class MSResEncoder(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        super().__init__()
        self.inplanes5_1 = 64
        self.inplanes5_2 = 64
        self.inplanes5_3 = 64

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer5x5_11 = self._make_layer5_1(BasicBlock5x5_1, 64, layers[0], stride=2)
        self.layer5x5_12 = self._make_layer5_1(BasicBlock5x5_1, 128, layers[1], stride=2)
        self.layer5x5_13 = self._make_layer5_1(BasicBlock5x5_1, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)
        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_1 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer5x5_21 = self._make_layer5_2(BasicBlock5x5_2, 64, layers[0], stride=2)
        self.layer5x5_22 = self._make_layer5_2(BasicBlock5x5_2, 128, layers[1], stride=2)
        self.layer5x5_23 = self._make_layer5_2(BasicBlock5x5_2, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_2 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer5x5_31 = self._make_layer5_3(BasicBlock5x5_3, 64, layers[0], stride=2)
        self.layer5x5_32 = self._make_layer5_3(BasicBlock5x5_3, 128, layers[1], stride=2)
        self.layer5x5_33 = self._make_layer5_3(BasicBlock5x5_3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool5_3 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(48384, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5_1(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_1, planes, stride, downsample))
        self.inplanes5_1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_1, planes))

        return nn.Sequential(*layers)

    def _make_layer5_2(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_2, planes, stride, downsample))
        self.inplanes5_2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_2, planes))

        return nn.Sequential(*layers)

    def _make_layer5_3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5_3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5_3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5_3, planes, stride, downsample))
        self.inplanes5_3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5_3, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer5x5_11(x0)
        x = self.layer5x5_12(x)
        x = self.layer5x5_13(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool5_1(x)

        y = self.layer5x5_21(x0)
        y = self.layer5x5_22(y)
        y = self.layer5x5_23(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5_2(y)

        z = self.layer5x5_31(x0)
        z = self.layer5x5_32(z)
        z = self.layer5x5_33(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool5_3(z)
        out = torch.cat([x, y, z], dim=1)
        return out


class MSResDecoder(nn.Module):
    def __init__(self, input_channel, layers=[256, 128, 64], num_classes=10):
        super().__init__()

        self.conv1 = nn.ConvTranspose1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def block(in_feat, out_feat, normalize=True):
            return nn.Sequential(nn.ConvTranspose1d(in_feat, in_feat, kernel_size=5, stride=1), nn.ConvTranspose1d(in_feat, out_feat, kernel_size=5, stride=2), nn.ReLU(), nn.ConvTranspose1d(out_feat, out_feat, kernel_size=5, stride=1))

        self.block1_1 = block(256, 128)
        self.block2_1 = block(128, 64)
        self.block3_1 = block(64, 64)
        self.block4_1 = block(64, 64)
        self.block5_1 = nn.Linear(1233, 1250)

        self.block1_2 = block(256, 128)
        self.block2_2 = block(128, 64)
        self.block3_2 = block(64, 64)
        self.block4_2 = block(64, 64)
        self.block5_2 = nn.Linear(1233, 1250)

        self.block1_3 = block(256, 128)
        self.block2_3 = block(128, 64)
        self.block3_3 = block(64, 64)
        self.block4_3 = block(64, 64)
        self.block5_3 = nn.Linear(1233, 1250)

        self.conv2 = nn.ConvTranspose1d(64 * 3, 64, kernel_size=1)
        self.conv1 = nn.ConvTranspose1d(64, input_channel, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        self.fc = nn.Linear(2499, 2500)

    def forward(self, x):
        x, y, z = x[:, :256], x[:, 256:256 + 256], x[:, 256 + 256:256 + 256 + 256]

        x = self.block1_1(x)
        x = self.block2_1(x)
        x = self.block3_1(x)
        x = self.block4_1(x)
        x = self.block5_1(x)

        y = self.block1_2(y)
        y = self.block2_2(y)
        y = self.block3_2(y)
        y = self.block4_2(y)
        y = self.block5_2(y)

        z = self.block1_3(z)
        z = self.block2_3(z)
        z = self.block3_3(z)
        z = self.block4_3(z)
        z = self.block5_3(z)

        cat_values = torch.cat([x, y, z], dim=1)
        x = self.conv2(cat_values)
        x = self.conv1(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MSResEncoder(1, num_classes=10)
    print(model)

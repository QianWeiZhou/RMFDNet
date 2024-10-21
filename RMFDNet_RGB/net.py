#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./res/resnet50-19c8e357.pth'), strict=False)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0, 0, 0, 0]):
        out0_10 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out0 = F.interpolate(out0_10, size=input1[1].size()[2:], mode='bilinear')
        out1_20 = F.relu(self.bn1(self.conv1(input1[1] + input2[1] + out0)), inplace=True)
        out1 = F.interpolate(out1_20, size=input1[2].size()[2:], mode='bilinear')
        out2_40 = F.relu(self.bn2(self.conv2(input1[2] + input2[2] + out1)), inplace=True)
        out2 = F.interpolate(out2_40, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3] + input2[3] + out2)), inplace=True)
        return out3, out2_40, out1_20, out0_10

    def initialize(self):
        weight_init(self)


class DecoderCR(nn.Module):  # 补偿和去余分支
    def __init__(self):
        super(DecoderCR, self).__init__()
        self.conv0 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input1):
        out0 = F.relu(self.bn0(self.conv0(input1[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1] + out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2] + out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3] + out2)), inplace=True)
        return out3

    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv1s = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1s = nn.BatchNorm2d(64)
        self.conv2s = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2s = nn.BatchNorm2d(64)
        self.conv3s = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3s = nn.BatchNorm2d(64)
        self.conv4s= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4s = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1s = F.relu(self.bn1s(self.conv1s(out1)), inplace=True)
        out2s = F.relu(self.bn2s(self.conv2s(out2)), inplace=True)
        out3s = F.relu(self.bn3s(self.conv3s(out3)), inplace=True)
        out4s = F.relu(self.bn4s(self.conv4s(out4)), inplace=True)


        return (out4s, out3s, out2s, out1s)

    def initialize(self):
        weight_init(self)


class LDF(nn.Module):
    def __init__(self, cfg):
        super(LDF, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet(cfg)
        self.conv5s = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4s = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3s = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2s = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4c = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3c = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2c = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5r = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4r = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3r = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2r = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder = Encoder()
        self.decoders = Decoder()
        self.decoderc = DecoderCR()
        self.decoderr = DecoderCR()

        self.linear = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        # self.linears = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                              nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))  # 补偿分支结果
        self.linearc = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))  # 补偿分支结果
        self.linearr = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))  # 多余分支结果
        self.outconv = nn.Conv2d(12, 1, 1)
        self.initialize()

    def forward(self, x, shape=None):
        out1o, out2, out3, out4, out5 = self.bkbone(x)
        out2s, out3s, out4s, out5s = self.conv2s(out2), self.conv3s(out3), self.conv4s(out4), self.conv5s(out5)
        out2c, out3c, out4c, out5c = self.conv2c(out2), self.conv3c(out3), self.conv4c(out4), self.conv5c(out5)
        out2r, out3r, out4r, out5r = self.conv2r(out2), self.conv3r(out3), self.conv4r(out4), self.conv5r(out5)
        del out1o, out2, out3, out4, out5

        outs1_80, outs1_40, outs1_20, outs1_10 = self.decoders([out5s, out4s, out3s, out2s])
        out2c1, out3c1, out4c1, out5c1 = torch.cat([out2c, outs1_80], dim=1), \
                                         torch.cat([out3c, outs1_40], dim=1), \
                                         torch.cat([out4c, outs1_20], dim=1), \
                                         torch.cat([out5c, outs1_10], dim=1)
        out2r1, out3r1, out4r1, out5r1 = torch.cat([out2r, outs1_80], dim=1), \
                                         torch.cat([out3r, outs1_40], dim=1), \
                                         torch.cat([out4r, outs1_20], dim=1), \
                                         torch.cat([out5r, outs1_10], dim=1)
        outcom1 = self.decoderc([out5c1, out4c1, out3c1, out2c1])
        outred1 = self.decoderr([out5r1, out4r1, out3r1, out2r1])
        #print(outb1_80.shape, outd1_80.shape,outcom1.shape,outred1.shape)
        out1 = torch.cat([outs1_80,outcom1,outred1], dim=1)
        del out2c1, out3c1, out4c1, out5c1,out2r1, out3r1, out4r1, out5r1

        outs2 = self.encoder(out1)
        outs2_80, outs2_40, outs2_20, outs2_10 = self.decoders([out5s, out4s, out3s, out2s], outs2)
        out2c2, out3c2, out4c2, out5c2 = torch.cat([out2c, outs2_80], dim=1),\
                                         torch.cat([out3c, outs2_40], dim=1), \
                                         torch.cat([out4c, outs2_20], dim=1), \
                                         torch.cat([out5c, outs2_10], dim=1)
        out2r2, out3r2, out4r2, out5r2 = torch.cat([out2r, outs2_80], dim=1), \
                                         torch.cat([out3r, outs2_40], dim=1), \
                                         torch.cat([out4r, outs2_20], dim=1), \
                                         torch.cat([out5r, outs2_10], dim=1)
        outcom2 = self.decoderc([out5c2, out4c2, out3c2, out2c2])
        outred2 = self.decoderr([out5r2, out4r2, out3r2, out2r2])
        out2 = torch.cat([outs2_80, outcom2,outred2], dim=1)
        del out2c2, out3c2, out4c2, out5c2,out2r2, out3r2, out4r2, out5r2

        outs3 = self.encoder(out2)
        outs3_80, outs3_40, outs3_20, outs3_10 = self.decoders([out5s, out4s, out3s, out2s], outs3)
        out2c3, out3c3, out4c3, out5c3 = torch.cat([out2c, outs3_80], dim=1), \
                                         torch.cat([out3c, outs3_40], dim=1), \
                                         torch.cat([out4c, outs3_20], dim=1), \
                                         torch.cat([out5c, outs3_10], dim=1)
        out2r3, out3r3, out4r3, out5r3 = torch.cat([out2r, outs3_80], dim=1), \
                                         torch.cat([out3r, outs3_40], dim=1), \
                                         torch.cat([out4r, outs3_20], dim=1), \
                                         torch.cat([out5r, outs3_10], dim=1)
        outcom3 = self.decoderc([out5c3, out4c3, out3c3, out2c3])
        outred3 = self.decoderr([out5r3, out4r3, out3r3, out2r3])
        out3 = torch.cat([outs3_80, outcom3, outred3], dim=1)
        del out2c3, out3c3, out4c3, out5c3,out2r3, out3r3, out4r3, out5r3


        outs4 = self.encoder(out3)
        outs4_80, outs4_40, outs4_20, outs4_10 = self.decoders([out5s, out4s, out3s, out2s], outs4)
        out2c4, out3c4, out4c4, out5c4 = torch.cat([out2c, outs4_80], dim=1), \
                                         torch.cat([out3c, outs4_40], dim=1), \
                                         torch.cat([out4c, outs4_20], dim=1), \
                                         torch.cat([out5c, outs4_10], dim=1)
        out2r4, out3r4, out4r4, out5r4 = torch.cat([out2r, outs4_80], dim=1), \
                                         torch.cat([out3r, outs4_40], dim=1), \
                                         torch.cat([out4r, outs4_20], dim=1), \
                                         torch.cat([out5r, outs4_10], dim=1)
        outcom4 = self.decoderc([out5c4, out4c4, out3c4, out2c4])
        outred4 = self.decoderr([out5r4, out4r4, out3r4, out2r4])
        out4 = torch.cat([outs4_80, outcom4, outred4], dim=1)
        del out2c4, out3c4, out4c4, out5c4,out2r4, out3r4, out4r4, out5r4

        if shape is None:
            shape = x.size()[2:]

        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        #outs1 = F.interpolate(self.linears(outs1_80), size=shape, mode='bilinear')
        outcom1 = F.interpolate(self.linearc(outcom1), size=shape, mode='bilinear')
        outred1 = F.interpolate(self.linearr(outred1), size=shape, mode='bilinear')


        out2 = F.interpolate(self.linear(out2), size=shape, mode='bilinear')
       # outs2 = F.interpolate(self.linears(outs2_80), size=shape, mode='bilinear')
        outcom2 = F.interpolate(self.linearc(outcom2), size=shape, mode='bilinear')
        outred2 = F.interpolate(self.linearr(outred2), size=shape, mode='bilinear')

        out3 = F.interpolate(self.linear(out3), size=shape, mode='bilinear')
        #outs3 = F.interpolate(self.linears(outs3_80), size=shape, mode='bilinear')
        outcom3 = F.interpolate(self.linearc(outcom3), size=shape, mode='bilinear')
        outred3 = F.interpolate(self.linearr(outred3), size=shape, mode='bilinear')

        out4 = F.interpolate(self.linear(out4), size=shape, mode='bilinear')
        #outs4 = F.interpolate(self.linears(outs4_80), size=shape, mode='bilinear')
        outcom4 = F.interpolate(self.linearc(outcom4), size=shape, mode='bilinear')
        outred4 = F.interpolate(self.linearr(outred4), size=shape, mode='bilinear')
        out = self.outconv(torch.cat((out1 ,outcom1, outred1, out2,outcom2, outred2, out3, outcom3, outred3, out4,outcom4, outred4), 1))
        return out1 ,outcom1, outred1, out2,outcom2, outred2, out3, outcom3, outred3, out4,outcom4, outred4,out

    def initialize(self):
        if self.cfg.snapshot:
            print("此时的snapshot={}".format(self.cfg.snapshot))
            self.load_state_dict(torch.load(self.cfg.snapshot)['net'])
        else:
            weight_init(self)
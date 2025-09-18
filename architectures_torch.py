import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )

        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


def make_complete_resnet(input_shape, num_classes=10):
    net = []

    net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
    net += [nn.BatchNorm2d(64)]
    net += [nn.ReLU()]
    net += [nn.MaxPool2d(2)]

    net += [ResBlock(64, 64)]
    net += [ResBlock(64, 128, stride=2)]
    net += [ResBlock(128, 128)]
    net += [ResBlock(128, 256, stride=2)]

    net += [nn.AdaptiveAvgPool2d(1)]
    net += [nn.Flatten()]
    net += [nn.Linear(256, num_classes)]

    return nn.Sequential(*net)


def split_resnet_client(input_shape, split_level, num_classes=10):
    complete_model = make_complete_resnet(input_shape, num_classes)

    if split_level == 1:
        client_layers = complete_model[:5]
    elif split_level == 2:
        client_layers = complete_model[:6]
    elif split_level == 3:
        client_layers = complete_model[:7]
    elif split_level == 4:
        client_layers = complete_model[:8]
    else:
        raise ValueError(f"Invalid split_level: {split_level}")

    return nn.Sequential(*client_layers)


def split_resnet_server(input_shape, split_level, num_classes=10):
    complete_model = make_complete_resnet(input_shape, num_classes)

    if split_level == 1:
        server_layers = complete_model[5:]
    elif split_level == 2:
        server_layers = complete_model[6:]
    elif split_level == 3:
        server_layers = complete_model[7:]
    elif split_level == 4:
        server_layers = complete_model[8:]
    else:
        raise ValueError(f"Invalid split_level: {split_level}")

    return nn.Sequential(*server_layers)


def pilot(input_shape, level):

    net = []

    act = None
    # act = 'swish'

    print("[PILOT] activation: ", act)

    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception("No level %d" % level)


def decoder(input_shape, level, channels=3):
    net = []

    # act = "relu"
    act = None

    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1)]

    if level == 1:
        net += [nn.Conv2d(256, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]

    if level <= 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(128, channels, 3, 2, 1, output_padding=1)]
    net += [nn.Tanh()]
    return nn.Sequential(*net)


def discriminator(input_shape, level):

    net = []
    if level == 1:
        net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape[0], 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]

    bn = False

    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(1024, 1)]
    return nn.Sequential(*net)


SETUPS = [
    (
        functools.partial(split_resnet_client, split_level=i),
        functools.partial(pilot, level=i),
        functools.partial(decoder, level=i),
        functools.partial(discriminator, level=i),
    )
    for i in range(1, 5)
]

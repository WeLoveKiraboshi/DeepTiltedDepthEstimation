import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.utils.model_zoo as model_zoo

import os
from models.PartialConv_v3 import PartialConv2d

import matplotlib.pyplot as plt
from matplotlib import cm

model_urls = {
    'pdresnet18': '',
    'pdresnet34': '',
    'pdresnet50': '',
    'pdresnet101': '',
    'pdresnet152': '',
}

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__()
        self.BatchNorm2d = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    def forward(self, input, mask):
        return self.BatchNorm2d(input), mask


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        x = inputs[0] #x
        mask = inputs[1] #mask
        for module in self._modules.values():
            #for tag, layer in module.named_parameters():
            # for layer in module.modules():
            #     print(layer)
            if type(inputs) == tuple:
                x, mask = module(x, mask)
            else:
                inputs = module(inputs)
        return x, mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.name = 'Bottleneck'

    def forward(self, x, mask):
        residual_x = x
        residual_mask = mask
        out, out_mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, out_mask = self.conv3(out, out_mask)
        out = self.bn3(out)

        if self.downsample is not None:
            residual_x, residual_mask = self.downsample(x, mask)

        out += residual_x
        out = self.relu(out)

        return out, residual_mask


def get_incoming_shape(incoming):
    size = incoming.size()
    # returns the incoming data shape as a list
    return [size[0], size[1], size[2], size[3]]

def interleave(tensors, axis):
    # change the first element (batch_size to -1)
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape

    # double 1 dimension
    new_shape[axis] *= len(tensors)

    # pack the tensors on top of each other
    stacked = torch.stack(tensors, axis+1)

    # reshape and return
    reshaped = stacked.view(new_shape)
    return reshaped

class UnpoolingAsConvolution(nn.Module):
    def __init__(self, inplanes, planes):
        super(UnpoolingAsConvolution, self).__init__()

        # interleaving convolutions
        self.conv_A = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_B = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(2, 3), stride=1, padding=0)
        self.conv_C = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 2), stride=1, padding=0)
        self.conv_D = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x):
        output_a = self.conv_A(x)

        padded_b = nn.functional.pad(x, (1, 1, 0, 1))
        output_b = self.conv_B(padded_b)

        padded_c = nn.functional.pad(x, (0, 1, 1, 1))
        output_c = self.conv_C(padded_c)

        padded_d = nn.functional.pad(x, (0, 1, 0, 1))
        output_d = self.conv_D(padded_d)

        left = interleave([output_a, output_b], axis=2)
        right = interleave([output_c, output_d], axis=2)
        y = interleave([left, right], axis=3)
        return y

class UpProjection(nn.Module):
    def __init__(self, inplanes, planes):
        super(UpProjection, self).__init__()

        self.unpool_main = UnpoolingAsConvolution(inplanes, planes)
        self.unpool_res = UnpoolingAsConvolution(inplanes, planes)

        self.main_branch = nn.Sequential(
            self.unpool_main,
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes)
        )

        self.residual_branch = nn.Sequential(
            self.unpool_res,
            nn.BatchNorm2d(planes),
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_data):
        x = self.main_branch(input_data)
        res = self.residual_branch(input_data)
        x += res
        x = self.relu(x)
        return x

class ConConv(nn.Module):
    def __init__(self, inplanes_x1, inplanes_x2, planes):
        super(ConConv, self).__init__()
        self.conv = nn.Conv2d(inplanes_x1 + inplanes_x2, planes, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)
        return x1


class ResnetUnetHybridPartialConv_v3(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], pretrained=False):
        self.inplanes = 64

        # resnet layers
        super(ResnetUnetHybridPartialConv_v3, self).__init__()
        self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # additional up projection layers parts
        self.conv2 = PartialConv2d(2048, 1024, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up_proj1 = UpProjection(1024, 512)
        self.up_proj2 = UpProjection(512, 256)
        self.up_proj3 = UpProjection(256, 128)
        self.up_proj4 = UpProjection(128, 64)

        self.drop = nn.Dropout(0.5, False)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

        # padding + concat for unet stuff
        self.con_conv1 = ConConv(1024, 1024, 512)
        self.con_conv2 = ConConv(512, 256, 256)
        self.con_conv3 = ConConv(256, 128, 128)
        self.con_conv4 = ConConv(64, 64, 1)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.network_name = 'ResnetUnetHybrid_pconv_v3'
        if pretrained:
            self.load_weight_resnet50()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MySequential(
                PartialConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MyBatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MySequential(*layers)


    def forward(self, x, mask):
        x, mask_to_conv4 = self.conv1(x, mask)
        x = self.bn1(x)
        x_to_conv4 = self.relu(x) #orch.Size([64, 64, 120, 160])
        x = self.maxpool(x_to_conv4)         #torch.Size([64, 64, 60, 80])
        mask_max_pool = self.maxpool(mask_to_conv4)  # torch.Size([2, 64, 60, 80])

        x_to_conv3, mask_to_conv3 = self.layer1(x, mask_max_pool)          #torch.Size([64, 256, 60, 80])
        x_to_conv2, mask_to_conv2 = self.layer2(x_to_conv3, mask_to_conv3)  # torch.Size([64, 512, 30, 40])
        x_to_conv1, mask_to_conv1 = self.layer3(x_to_conv2, mask_to_conv2)  # torch.Size([64, 1024, 15, 20])
        x, mask_layer4 = self.layer4(x_to_conv1, mask_to_conv1)  # torch.Size([64, 2048, 15, 20])

        # additional layers
        x, mask_conv2 = self.conv2(x, mask_layer4)                 #torch.Size([64, 1024, 15, 20])
        x = self.bn2(x)                   #torch.Size([64, 1024, 15, 20]

        # # up project part
        x = self.con_conv1(x, x_to_conv1)

        x = self.up_proj2(x)
        x = self.con_conv2(x, x_to_conv2)

        x = self.up_proj3(x)
        x = self.con_conv3(x, x_to_conv3)

        x = self.up_proj4(x)
        x = self.con_conv4(x, x_to_conv4)

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x, mask_conv2


    def load_weight_resnet50(self):
        resnet50 = torchvision.models.resnet50(pretrained=True) #model_zoo.load_url(model_urls['pdresnet50'])
        own_state = self.state_dict()
        resnet50_state = resnet50.state_dict()
        for name, param in resnet50_state.items():
            if name not in own_state:
                #print('layer not exist... {} {}'.format(name, param.shape)) # fc.weight & fc.bs
                continue
            else:
                #print('load weight... {}'.format(name))
                if isinstance(param, torch.nn.parameter.Parameter):
                    param = param.data
                own_state[name].copy_(param)

        own_state['layer1.0.downsample.1.BatchNorm2d.weight'].copy_(resnet50_state['layer1.0.downsample.1.weight'])
        own_state['layer1.0.downsample.1.BatchNorm2d.bias'].copy_(resnet50_state['layer1.0.downsample.1.bias'])
        own_state['layer1.0.downsample.1.BatchNorm2d.running_mean'].copy_(
            resnet50_state['layer1.0.downsample.1.running_mean'])
        own_state['layer1.0.downsample.1.BatchNorm2d.running_var'].copy_(
            resnet50_state['layer1.0.downsample.1.running_var'])
        own_state['layer1.0.downsample.1.BatchNorm2d.num_batches_tracked'].copy_(
            resnet50_state['layer1.0.downsample.1.num_batches_tracked'])
        own_state['layer2.0.downsample.1.BatchNorm2d.weight'].copy_(resnet50_state['layer2.0.downsample.1.weight'])
        own_state['layer2.0.downsample.1.BatchNorm2d.bias'].copy_(resnet50_state['layer2.0.downsample.1.bias'])
        own_state['layer2.0.downsample.1.BatchNorm2d.running_mean'].copy_(
            resnet50_state['layer2.0.downsample.1.running_mean'])
        own_state['layer2.0.downsample.1.BatchNorm2d.running_var'].copy_(
            resnet50_state['layer2.0.downsample.1.running_var'])
        own_state['layer2.0.downsample.1.BatchNorm2d.num_batches_tracked'].copy_(
            resnet50_state['layer2.0.downsample.1.num_batches_tracked'])
        own_state['layer3.0.downsample.1.BatchNorm2d.weight'].copy_(resnet50_state['layer3.0.downsample.1.weight'])
        own_state['layer3.0.downsample.1.BatchNorm2d.bias'].copy_(resnet50_state['layer3.0.downsample.1.bias'])
        own_state['layer3.0.downsample.1.BatchNorm2d.running_mean'].copy_(
            resnet50_state['layer3.0.downsample.1.running_mean'])
        own_state['layer3.0.downsample.1.BatchNorm2d.running_var'].copy_(
            resnet50_state['layer3.0.downsample.1.running_var'])
        own_state['layer3.0.downsample.1.BatchNorm2d.num_batches_tracked'].copy_(
            resnet50_state['layer3.0.downsample.1.num_batches_tracked'])
        own_state['layer4.0.downsample.1.BatchNorm2d.weight'].copy_(resnet50_state['layer4.0.downsample.1.weight'])
        own_state['layer4.0.downsample.1.BatchNorm2d.bias'].copy_(resnet50_state['layer4.0.downsample.1.bias'])
        own_state['layer4.0.downsample.1.BatchNorm2d.running_mean'].copy_(
            resnet50_state['layer4.0.downsample.1.running_mean'])
        own_state['layer4.0.downsample.1.BatchNorm2d.running_var'].copy_(
            resnet50_state['layer4.0.downsample.1.running_var'])
        own_state['layer4.0.downsample.1.BatchNorm2d.num_batches_tracked'].copy_(
            resnet50_state['layer4.0.downsample.1.num_batches_tracked'])
        print('loaded resnet-50 pretrained weight in {}'.format(self.network_name))
        return self







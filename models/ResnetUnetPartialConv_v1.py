import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from models.PartialConv import PartialConv, PartialConvTransposed2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 conv layer with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def visualize_mask(mask, layer='conv1'):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    print('{} : {}'.format(layer, mask.shape))  # [bs, 3, 240, 320] -> , [bs, 64, 120, 160]
    img = mask[0].detach().cpu().permute(1, 2, 0)[:, :, 0]
    plt.imshow(img)
    plt.show()

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__()
        self.BatchNorm2d = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    def forward(self, input, mask):
        return self.BatchNorm2d(input), mask

class MyReLU(nn.Module):
    def __init__(self,  inplace = False):
        super(MyReLU, self).__init__()
        self.activate = nn.ReLU(inplace=inplace)
    def forward(self, input, mask):
        return self.activate(input), mask


class mySequential(nn.Sequential):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv(planes, planes * 4, kernel_size=1, bias=False)
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
        self.conv_A = PartialConv(in_channels=inplanes, out_channels=planes, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_B = PartialConv(in_channels=inplanes, out_channels=planes, kernel_size=(2, 3), stride=1, padding=0)
        self.conv_C = PartialConv(in_channels=inplanes, out_channels=planes, kernel_size=(3, 2), stride=1, padding=0)
        self.conv_D = PartialConv(in_channels=inplanes, out_channels=planes, kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x, mask):
        output_a, output_a_mask = self.conv_A(x, mask)

        padded_b = nn.functional.pad(x, (1, 1, 0, 1))
        padded_b_mask = nn.functional.pad(mask, (1, 1, 0, 1))
        output_b, output_b_mask = self.conv_B(padded_b, padded_b_mask)

        padded_c = nn.functional.pad(x, (0, 1, 1, 1))
        padded_c_mask = nn.functional.pad(mask, (0, 1, 1, 1))
        output_c, output_c_mask = self.conv_C(padded_c, padded_c_mask)

        padded_d = nn.functional.pad(x, (0, 1, 0, 1))
        padded_d_mask = nn.functional.pad(mask, (0, 1, 0, 1))
        output_d, output_d_mask = self.conv_D(padded_d,padded_d_mask)

        #print('a = ', output_a.shape)
        #print('b = ', output_b.shape)
        left = interleave([output_a, output_b], axis=2)
        #print('left = ', left.shape)
        right = interleave([output_c, output_d], axis=2)
        #print('right = ', right.shape)
        y = interleave([left, right], axis=3)
        #print('y = ', y.shape)
        left_mask = interleave([output_a_mask, output_b_mask], axis=2)
        right_mask = interleave([output_c_mask, output_d_mask], axis=2)
        y_mask = interleave([left_mask, right_mask], axis=3)
        #print('y_mask = ', y_mask.shape)
        return y, y_mask

class UpProjection(nn.Module):
    def __init__(self, inplanes, planes):
        super(UpProjection, self).__init__()

        self.unpool_main = UnpoolingAsConvolution(inplanes, planes)
        self.unpool_res = UnpoolingAsConvolution(inplanes, planes)

        self.main_branch = mySequential(
            self.unpool_main,
            MyBatchNorm2d(planes),
            MyReLU(inplace=False),
            PartialConv(planes, planes, kernel_size=3, stride=1, padding=1),
            MyBatchNorm2d(planes)
        )

        self.residual_branch = mySequential(
            self.unpool_res,
            MyBatchNorm2d(planes),
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_data, input_mask):
        x, x_mask = self.main_branch(input_data, input_mask)
        res, res_mask = self.residual_branch(input_data, input_mask)
        x += res
        x = self.relu(x)
        return x, res_mask

class ConConv(nn.Module):
    def __init__(self, inplanes_x1, inplanes_x2, planes):
        super(ConConv, self).__init__()
        self.conv = PartialConv(inplanes_x1 + inplanes_x2, planes, kernel_size=1, bias=True)

    def forward(self, x1, x2, x1_mask, x2_mask):
        # print('x1 = {}, x2 = {}'.format(x1.shape, x2.shape))
        # print('x1 mask = {}, x2 mask = {}'.format(x1_mask.shape, x2_mask.shape))
        x1 = torch.cat([x2, x1], dim=1)
        x1_mask = torch.cat([x2_mask, x1_mask], dim=1)
        # print('after concat = ', x1.shape)
        # print('after mask concat = ', x1_mask.shape)
        x1, x1_mask = self.conv(x1, x1_mask)
        return x1, x1_mask

class downsample(nn.Module):
    def  __init__(self, inplanes=64, planes=64, block=Bottleneck, stride=1):
        super(downsample, self).__init__()
        self.pconv1 = PartialConv(inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * block.expansion)

    def forward(self, x, mask):
        x, mask = self.pconv1(x, mask)
        x = self.bn1(x)
        return x, mask

class my_make_layer(nn.Module):
    def __init__(self, outer_class_instance, block, planes, blocks, stride=1):
        super(my_make_layer, self).__init__()

        self.block = block
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        self.outer_class_instance = outer_class_instance
        #self.prev_inplanes = self.outer_class_instance.inplanes

        if self.stride != 1 or self.outer_class_instance != self.planes * self.block.expansion:
            self.downsample = downsample(self.outer_class_instance.inplanes, self.planes, self.block, self.stride)
        else:
            self.downsample = None

        self.block1 = block(self.outer_class_instance.inplanes, self.planes, self.stride, self.downsample)
        self.outer_class_instance.inplanes = self.planes * self.block.expansion

        self.loop_block1 = self.block(self.outer_class_instance.inplanes, self.planes)
        self.loop_block2 = self.block(self.outer_class_instance.inplanes, self.planes)
        self.loop_block3 = self.block(self.outer_class_instance.inplanes, self.planes)
        if self.blocks == 4:
            self.loop_block4 = self.block(self.outer_class_instance.inplanes, self.planes)
        elif self.blocks == 6:
            self.loop_block4 = self.block(self.outer_class_instance.inplanes, self.planes)
            self.loop_block5 = self.block(self.outer_class_instance.inplanes, self.planes)
            self.loop_block6 = self.block(self.outer_class_instance.inplanes, self.planes)
        # for i in range(1, self.blocks): #layers=[3, 4, 6, 3]
        #     self.layers.append(self.block(self.outer_class_instance.inplanes, self.planes))

    def forward(self, x, mask):
        if self.stride != 1 or self.prev_inplanes != self.planes * self.block.expansion:
            x, mask = self.pconv1(x, mask)
            x = self.bn1(x)
        x, mask = self.block1(x, mask)
        x, mask = self.loop_block1(x, mask)
        x, mask = self.loop_block2(x, mask)
        x, mask = self.loop_block3(x, mask)
        if self.block == 4:
            x, mask = self.loop_block4(x, mask)
        elif self.block == 6:
            x, mask = self.loop_block4(x, mask)
            x, mask = self.loop_block5(x, mask)
            x, mask = self.loop_block6(x, mask)
        return x, mask


class ResnetUnetHybridPartialConv_v1(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64

        # resnet layers
        super(ResnetUnetHybridPartialConv_v1, self).__init__()
        self.conv1 = PartialConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # additional up projection layers parts
        self.conv2 = PartialConv(2048, 1024, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up_proj1 = UpProjection(1024, 512)
        self.up_proj2 = UpProjection(512, 256)
        self.up_proj3 = UpProjection(256, 128)
        self.up_proj4 = UpProjection(128, 64)

        #self.drop = nn.Dropout(0.5, False)
        #self.deconv3 = F.interpolate(h_mask, scale_factor=2, mode='nearest')
        #self.deconv3 = PartialConvTransposed2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#
        # padding + concat for unet stuff
        # self.con_conv1 = ConConv(1024, 512, 512)
        # self.con_conv2 = ConConv(512, 256, 256)
        # self.con_conv3 = ConConv(256, 128, 128)
        # self.con_conv4 = ConConv(64, 64, 64)

        self.con_conv1 = ConConv(1024, 1024, 512)
        self.con_conv2 = ConConv(512, 256, 256)
        self.con_conv3 = ConConv(256, 128, 128)
        self.con_conv4 = ConConv(64, 64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = mySequential(
                PartialConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MyBatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return mySequential(*layers)


    def forward(self, x, mask):
        # visualize_mask(mask, layer='conv1')
        x, mask_to_conv4 = self.conv1(x, mask)
        x = self.bn1(x)
        x_to_conv4 = self.relu(x) #orch.Size([64, 64, 120, 160])


        x = self.maxpool(x_to_conv4)         #torch.Size([64, 64, 60, 80])
        mask = self.maxpool(mask_to_conv4)  #torch.Size([2, 64, 60, 80])
        # visualize_mask(mask, layer='maxpool')

        x_to_conv3, mask_to_conv3 = self.layer1(x, mask)          #torch.Size([64, 256, 60, 80])
        #visualize_mask(mask_to_conv3, layer='layer1')
        x_to_conv2, mask_to_conv2 = self.layer2(x_to_conv3, mask_to_conv3) #torch.Size([64, 512, 30, 40])
        x_to_conv1, mask_to_conv1 = self.layer3(x_to_conv2, mask_to_conv2) #torch.Size([64, 1024, 15, 20])
        x, mask = self.layer4(x_to_conv1, mask_to_conv1) #torch.Size([64, 2048, 15, 20])
        # visualize_mask(mask, layer='Layer4')

        # additional layers
        x, mask = self.conv2(x, mask)                 #torch.Size([64, 1024, 15, 20])
        x = self.bn2(x)                               #torch.Size([64, 1024, 15, 20]

        # # up project part
        x, mask = self.con_conv1(x, x_to_conv1, mask, mask_to_conv1)

        x, mask = self.up_proj2(x, mask)
        x, mask = self.con_conv2(x, x_to_conv2, mask, mask_to_conv2)
        # visualize_mask(mask, layer='Decode conv2')

        x, mask = self.up_proj3(x, mask)
        x, mask = self.con_conv3(x, x_to_conv3, mask, mask_to_conv3)

        x, mask = self.up_proj4(x, mask)
        x, mask = self.con_conv4(x, x_to_conv4, mask, mask_to_conv4)
        # visualize_mask(mask, layer='Decode conv4')

        #x = self.drop(x)
        #x, mask = self.deconv3(x, mask)
        #x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        return x, mask


def load_pretrained(model):
    import torchvision
    resnet18 = torchvision.models.resnet18(pretrained=True)

    own_state = model.state_dict()
    for name, param in resnet18.state_dict().items():
        if name not in own_state:
             print('layer not exist... {}'.format(name))
             continue
             own_state['conv1.input_conv.weight'].copy_(resnet18.state_dict()['conv1.weight'])
             own_state['conv1.input_conv.weight'].copy_(resnet18.state_dict()['conv1.weight'])
        else:
            if isinstance(param, torch.nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)


# layer not exist... conv1.weight
# layer not exist... layer1.0.conv1.weight
# layer not exist... layer1.0.conv2.weight
# layer not exist... layer1.1.conv1.weight
# layer not exist... layer1.1.conv2.weight
# layer not exist... layer2.0.conv1.weight
# layer not exist... layer2.0.conv2.weight
# layer not exist... layer2.0.downsample.0.weight
# layer not exist... layer2.0.downsample.1.weight
# layer not exist... layer2.0.downsample.1.bias
# layer not exist... layer2.0.downsample.1.running_mean
# layer not exist... layer2.0.downsample.1.running_var
# layer not exist... layer2.0.downsample.1.num_batches_tracked
# layer not exist... layer2.1.conv1.weight
# layer not exist... layer2.1.conv2.weight
# layer not exist... layer3.0.conv1.weight
# layer not exist... layer3.0.conv2.weight
# layer not exist... layer3.0.downsample.0.weight
# layer not exist... layer3.0.downsample.1.weight
# layer not exist... layer3.0.downsample.1.bias
# layer not exist... layer3.0.downsample.1.running_mean
# layer not exist... layer3.0.downsample.1.running_var
# layer not exist... layer3.0.downsample.1.num_batches_tracked
# layer not exist... layer3.1.conv1.weight
# layer not exist... layer3.1.conv2.weight
# layer not exist... layer4.0.conv1.weight
# layer not exist... layer4.0.conv2.weight
# layer not exist... layer4.0.downsample.0.weight
# layer not exist... layer4.0.downsample.1.weight
# layer not exist... layer4.0.downsample.1.bias
# layer not exist... layer4.0.downsample.1.running_mean
# layer not exist... layer4.0.downsample.1.running_var
# layer not exist... layer4.0.downsample.1.num_batches_tracked
# layer not exist... layer4.1.conv1.weight
# layer not exist... layer4.1.conv2.weight
# layer not exist... fc.weight
# layer not exist... fc.bias

###########################################################################################
# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py #
# Mainly changed the model forward() function                                             #
###########################################################################################

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal



try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dohisto=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dohisto = dohisto

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # if self.dohisto:
        #     temp = out
        #     print(np.count_nonzero(temp.cpu().detach().numpy()))
        #     print(torch.max(out), torch.min(out))

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # if self.dohisto:
        #     print(torch.max(out), torch.min(out))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # if self.dohisto:
        #     temp = identity
        #     print(np.count_nonzero(temp.cpu().detach().numpy()))
        #     print(torch.max(out), torch.min(out))

        return out

class ResNet(nn.Module):
    temp = 30
    def testhook(self, module, input, output):
        # print(torch.max(input[0]), torch.min(input[0]))
        # print(output[0])
        self._get_histo(output[0], self.temp)
        self.temp+=1


    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, clip_range=None, aggregation = 'mean', 
                 dohisto=False, collapsefunc=None):
        super(ResNet, self).__init__()
        self.i = 0
        self.clip_range = clip_range
        self.aggregation = aggregation
       
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                        dohisto =dohisto)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       dohisto = dohisto)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       dohisto = dohisto)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       dohisto = dohisto)

        # for i in range(len(self.layer1)):
        #     self.layer1[i].relu.register_forward_hook(self.testhook)
        # for i in range(len(self.layer2)):
        #     self.layer2[i].relu.register_forward_hook(self.testhook)
        # for i in range(len(self.layer3)):
        #     self.layer3[i].relu.register_forward_hook(self.testhook)
        # for i in range(len(self.layer4)):
        #     self.layer4[i].relu.register_forward_hook(self.testhook)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dohisto = dohisto
        self.collapsefunc = collapsefunc

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dohisto = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dohisto=dohisto))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # get a histo of all features, flattened through the batch
    def _get_histo(self, x, layer):
        if self.dohisto:
            pass
            # print(torch.mean(x))
            # flat = torch.flatten(x).cpu().detach().numpy()
            # # print(np.max(flat), np.min(flat))
            # n, bins = np.histogram(flat, bins=50)
            
            # bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
            # plt.plot(bins_mean, n, label='layer'+str(layer))

            
            # out = x.cpu().detach().numpy()
            # out = np.linalg.norm(out, axis=1).flatten()
            # print(out.shape)
            # plt.figure()
            # _ = plt.hist(out)
            # plt.savefig(name)
            
            batch = x.cpu().detach().numpy()       
            for i in range(len(x)):
                out = batch[i]
                if layer==-1: 
                    out = np.transpose(out, (1,2,0))
                else: 
                    out = np.max(out, axis=0)
                plt.figure()
                plt.imshow(out)
                plt.savefig(str(i) + '_adp_atk_layer_' + str(layer))
                plt.close()

            
    def _mask(self, x, mean, stddev, patchsizes):
        # collapse in C
        temp = x.cpu().detach().numpy()
        mean_ = mean.cpu().detach().numpy()
        stddev_ = stddev.cpu().detach().numpy()
        # print(mean, stddev)
        
        if self.collapsefunc == 'max':
            collapsed = np.max(temp, axis=1)
            # print('maxed', collapsed.shape)
            mean_ = np.max(mean_)
            stddev_ = np.max(stddev_)
        elif self.collapsefunc == 'l2':
            collapsed = np.linalg.norm(temp, axis=1)
            # print('l2ed', collapsed.shape)
            mean_ = np.linalg.norm(mean_)
            stddev_ = np.linalg.norm(stddev_)
        else: 
            return x

        for i in range(len(collapsed)):
            max_=-1
            r,c = 0,0
            size = patchsizes[0]
            for s in range(patchsizes[0], patchsizes[1]):
                f = np.ones((s,s,))/ (s) #note not normalized
                smoothed = scipy.signal.convolve2d(collapsed[i,:,:], f, mode='valid')
                curr_max = smoothed.max()
                if curr_max > max_:
                    max_ = curr_max 
                    r,c, = np.unravel_index(smoothed.argmax(), smoothed.shape)
                    size = s
            # TODO: set threshold/probablistic mask
            # /s is due to the values not being normalized
            if max_/size > mean_+stddev_*2:  
                mask = torch.zeros(x.shape).to('cuda')
                mask[i,:, r:(r+size), c:(c+size)] = 1
                x = torch.where(mask == 1, torch.tensor(0.).to('cuda'), x) 
        
        return x
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.shape)
        self._get_histo(x, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self._mask(x, self.bn1.bias, self.bn1.weight, (18,26)) 
        x = self.maxpool(x)

        # if self.dohisto:
        #     plt.figure()
            # plt.semilogy()
        
        self._get_histo(x, 0)

        x = self.layer1(x)
        # zero = torch.zeros(x.size()).to('cuda')
        # x = torch.where(x < 1.0, x, zero)
        # x = torch.clamp(x, 0, 2.5)
        # print('layer1', self.layer1[2].bn3.bias, self.layer1[2].bn3.weight)
        # print(x.shape)
        x = self._mask(x, torch.add(self.layer1[2].bn3.bias, self.layer1[1].bn3.bias), 
            torch.sqrt(torch.add(torch.pow(self.layer1[2].bn3.weight, 2), torch.pow(self.layer1[1].bn3.weight, 2))),
            (8, 15))

        self._get_histo(x, 1)
        x = self.layer2(x)
        # zero = torch.zeros(x.size()).to('cuda')
        # x = torch.where(x < 1, x, zero)
        # x = torch.clamp(x, 0, 1.5)
        # print(x.shape)
        x = self._mask(x, torch.add(self.layer2[3].bn3.bias, self.layer2[2].bn3.bias), 
            torch.sqrt(torch.add(torch.pow(self.layer2[3].bn3.weight, 2), torch.pow(self.layer2[2].bn3.weight, 2))),
            (3,10))

        self._get_histo(x, 2)
        x = self.layer3(x)
        # zero = torch.zeros(x.size()).to('cuda')
        # x = torch.where(x < 1, x, zero)
        # x = torch.clamp(x, 0, 2)
        # print(x.shape)
        x = self._mask(x, torch.add(self.layer3[5].bn3.bias, self.layer3[4].bn3.bias), 
            torch.sqrt(torch.add(torch.pow(self.layer3[5].bn3.weight, 2), torch.pow(self.layer3[4].bn3.weight, 2))),
            (1, 5))
        self._get_histo(x, 3)
        x = self.layer4(x)
        # print(x.shape)
        # zero = torch.zeros(x.size()).to('cuda')
        # x = torch.where(x < 1, x, zero)
        # x = torch.clamp(x, 0, 1)
        # print(x.shape)
        x = self._mask(x, torch.add(self.layer4[2].bn3.bias, self.layer4[1].bn3.bias), 
            torch.sqrt(torch.add(torch.pow(self.layer4[2].bn3.weight, 2), torch.pow(self.layer4[1].bn3.weight, 2))), 
            (1,3))
        self._get_histo(x, 4)

        # if self.dohisto:
        #     s = 'b_zeros' + str(self.i)
        #     # self.i+=1
        #     plt.title(s)
        #     plt.legend()
        #     plt.savefig(s)

        x = x.permute(0,2,3,1)
        x = self.fc(x)
        if self.clip_range is not None:
            x = torch.clamp(x,self.clip_range[0],self.clip_range[1])
        if self.aggregation == 'mean':
            x = torch.mean(x,dim=(1,2))
        elif self.aggregation == 'median':
            x = x.view([x.size()[0],-1,10])
            x = torch.median(x,dim=1)
            return x.values
        elif self.aggregation =='cbn': # clipping function from Clipped BagNet
            x = torch.tanh(x*0.05-1)
            x = torch.mean(x,dim=(1,2))
        elif self.aggregation == 'none':
            pass
        # print(x.shape)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
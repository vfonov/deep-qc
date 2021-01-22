import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from torchvision import models
from torch.nn.parameter import Parameter

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
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

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetQC(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
	use_ref:bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetQC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.use_ref = use_ref
        self.feat = 3
        self.inplanes = 64
        self.dilation = 1
        self.expansion = block.expansion
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(2 if self.use_ref else 1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for merging multiple features
        self.addon = nn.Sequential(
            nn.Conv2d(self.feat * 512 * block.expansion, 512*block.expansion, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512*block.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(512*block.expansion, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5,inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        # initialization
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
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        # split feats into batches
        x = x.view(-1, 2 if self.use_ref else 1 ,224,224)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # merge batches together
        x = x.view(-1, 512*self.feat * self.expansion, 7, 7)
        x = self.addon(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


    def load_from_std(self, std_model):
        # import weights from the standard ResNet model
        # TODO: finish
        # first load all standard items
        own_state = self.state_dict()
        for name, param in std_model.state_dict().items():
            if name == 'conv1.weight':
                if isinstance(param, Parameter):
                    param = param.data
                # convert to mono weight
                # collaps parameters along second dimension, emulating grayscale feature 
                mono_param=param.sum( 1, keepdim=True )
                if self.use_ref:
                    own_state[name].copy_( torch.cat((mono_param,mono_param),1) )
                else:
                    own_state[name].copy_( mono_param )
                pass
            elif name == 'fc.weight' or name == 'fc.bias' or name == 'conv2.weight' or name == 'conv2.bias':
                # don't use at all
                pass
            elif name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            


def _resnet_qc(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    progress: bool,
    **kwargs: Any
) -> ResNetQC:
    return ResNetQC(block, layers, **kwargs)


def resnet_qc_18(pretrained: bool=False, progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet_qc( BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet18(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_34(pretrained=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc(BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet34(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_50(pretrained: bool=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc('resnet50', Bottleneck, [3, 4, 6, 3], progress, **kwargs)

    if pretrained:
        model_ft = models.resnet50(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_101(pretrained: bool=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc( Bottleneck, [3, 4, 23, 3], progress,
                   **kwargs)
    if pretrained:
        model_ft = models.resnet101(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_152(pretrained: bool=False, progress: bool = True, **kwargs)  -> ResNetQC:
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc( Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

    if pretrained:
        model_ft = models.resnet152(pretrained=True)
        model.load_from_std(model_ft)
    return model



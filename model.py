#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)

class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

class MobileShuffleUnit(nn.Module):
    def __init__(self, channels, stride, num_conv_branches, use_se, inference_mode):
        super(MobileShuffleUnit, self).__init__()
        self.stride = stride
        self.channels = channels
        self.num_conv_branches = num_conv_branches
        self.inference_mode = inference_mode

        self.residual = [] # 左边的支路
        self.shortcut = [] # 右边的支路
        self.residual.append(MobileShuffleBlock(in_channels=self.channels,
                                        out_channels=self.channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=use_se,
                                        num_conv_branches=self.num_conv_branches))
        self.residual.append(MobileShuffleBlock(in_channels=self.channels,
                                    out_channels=self.channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=self.channels,
                                    inference_mode=self.inference_mode,
                                    use_se=use_se,
                                    num_conv_branches=self.num_conv_branches))
        self.residual.append(MobileShuffleBlock(in_channels=self.channels,
                                        out_channels=self.channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=use_se,
                                        num_conv_branches=self.num_conv_branches))
        if stride == 2:
            self.shortcut.append(MobileShuffleBlock(in_channels=self.channels,
                                        out_channels=self.channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=self.channels,
                                        inference_mode=self.inference_mode,
                                        use_se=use_se,
                                        num_conv_branches=self.num_conv_branches))
            self.shortcut.append(MobileShuffleBlock(in_channels=self.channels,
                                        out_channels=self.channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=use_se,
                                        num_conv_branches=self.num_conv_branches))
        self.residual = nn.Sequential(*self.residual) # Sequential内部自带forward方法
        self.shortcut = nn.Sequential(*self.shortcut)
    def forward(self,x):
        if self.stride == 2:
            residual = shortcut = x
        else:
            residual, shortcut = channel_split(x, self.channels)
        residual = self.residual(residual)
        shortcut = self.shortcut(shortcut)
        return channel_shuffle(torch.cat([shortcut, residual], dim=1), 2)

class MobileShuffleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv) # 用ModuleList把list[Sequential]包装起来,
            # 这样在后续重写forward时，通过索引来调用里面的Sequential

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x) # 在此处通过索引调用ModuleList内部的Sequential

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias() # 将rbr_conv,rbr_scale,rbr_skip相加后的kernel和bias
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix]) # 得到每个branch融合BN后的kernel和bias
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential): # rbr_conv与rbr_scale的处理方式
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d) # rbr_skip的处理方式
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class MobileShuffle(nn.Module):
    def __init__(self,
                num_classes: int = 1000,
                inference_mode: bool = False,
                width_multipliers: Optional[List[float]] = None,
                num_conv_branches: int = 1,
                num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                use_se: bool = False,
                ) -> None:
        super().__init__() # 这里如果用super(MobileShuffle)会报错，初始化失败，py3无需传入类名，自动继承
        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = int(24 * width_multipliers[0])
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileShuffleBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode,
                                     use_se=False,
                                     num_conv_branches=self.num_conv_branches)
        self.stage1 = self._make_stage(int(24 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(48 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(96 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(192 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5 = MobileShuffleBlock(in_channels=int(192 * width_multipliers[3] * 2), out_channels=1024,
                                     kernel_size=1, stride=1, padding=0,
                                     inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches)
        self.linear = nn.Linear(1024, num_classes)
    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        Units = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True
            Units.append(
                MobileShuffleUnit(
                planes, stride, self.num_conv_branches, use_se, self.inference_mode
                ))
        return nn.Sequential(*Units) # 用nn.Sequential把MobileShuffleUnit包起来，方便直接调用,Sequential内部是顺序调用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# def mobileshuffle(num_classes: int = 1000, inference_mode: bool = False,
#               variant: str = "s0") -> nn.Module:
#     PARAMS = {
#         "s0": {"width_multipliers": (1.0, 1.0, 1.0, 1.0),
#                "num_conv_branches": 4, "num_blocks_per_stage": [2, 8, 10, 1],
#                "use_se": True},
#         "s1": {"width_multipliers": (1.5, 1.5, 1.5, 1.5),
#                "num_conv_branches": 4, "num_blocks_per_stage": [2, 4, 6, 1],
#                "use_se": True},
#         "s2": {"width_multipliers": (2.0, 2.0, 2.0, 2.0),
#                "num_conv_branches": 4, "num_blocks_per_stage": [2, 4, 6, 1],
#                "use_se": True},
#         "s3": {"width_multipliers": (3.0, 3.0, 3.0, 3.0),
#                "num_conv_branches": 4, "num_blocks_per_stage": [2, 4, 6, 1],
#                "use_se": True},
#         "s4": {"width_multipliers": (4.0, 4.0, 4.0, 4.0),
#                "num_conv_branches": 4, "num_blocks_per_stage": [2, 4, 6, 1],
#                "use_se": True},
#     }
#     variant_params = PARAMS[variant]
#     return MobileShuffle(num_classes=num_classes,
#                          inference_mode=inference_mode,
#                         **variant_params)
#

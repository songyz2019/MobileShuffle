#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']

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


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
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
                                          dilation=dilation,
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
            self.rbr_conv = nn.ModuleList(rbr_conv)

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
            out += self.rbr_conv[ix](x)

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
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
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


class MobileOne(nn.Module):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 # num_blocks_per_stage: List[int] = [4,7,4,1],
                num_blocks_per_stage: List[int] = [2,8,10,1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = True,
                 num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(24, int(24 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode,
                                     num_conv_branches=self.num_conv_branches)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,dilation=1,ceil_mode=False)
        self.cur_layer_idx = 1
        # 24 48 96 192
        # 32 64 128 256
        # 48 96 192 384
        self.stage1 = self._make_stage(int(24 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(48 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(96 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(192 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(384, 1024, 1, stride=1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        self.conv5 = MobileOneBlock(in_channels=384, out_channels=1024,
                                     kernel_size=1, stride=1, padding=0,
                                     inference_mode=self.inference_mode,
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
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # 新增加的PW卷积
            blocks.append(MobileOneBlock(in_channels=planes,
                                        out_channels=planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=use_se,
                                        num_conv_branches=self.num_conv_branches))


            # self.in_planes = planes
            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=planes,
                                         out_channels=planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))

            if stride == 2:

                blocks.append(MobileOneBlock(in_channels=planes,
                                             out_channels=planes,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=1,
                                             groups=self.in_planes,
                                             inference_mode=self.inference_mode,
                                             use_se=use_se,
                                             num_conv_branches=self.num_conv_branches))

                blocks.append(MobileOneBlock(in_channels=planes,
                                             out_channels=planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             inference_mode=self.inference_mode,
                                             use_se=use_se,
                                             num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)
        # x = self.maxpool(x)
        "stage1的4个block"
        shortcut = x
        residual = x
        stage1_0_residual = self.stage1[:3]
        stage1_0_shortcut = self.stage1[3:5]
        residual = stage1_0_residual(residual)
        shortcut = stage1_0_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        stage1_channel = 24
        shortcut, residual = channel_split(x,stage1_channel)
        stage1_1_residual = self.stage1[5:8]
        stage1_1_shortcut = nn.Identity()
        residual = stage1_1_residual(residual)
        shortcut = stage1_1_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        # shortcut, residual = channel_split(x,stage1_channel)
        # stage1_2_residual = self.stage1[8:11]
        # stage1_2_shortcut = nn.Identity()
        # residual = stage1_2_residual(residual)
        # shortcut = stage1_2_shortcut(shortcut)
        # x = torch.cat([shortcut, residual], dim=1)
        # x = channel_shuffle(x, 2)
        #
        # shortcut, residual = channel_split(x,stage1_channel)
        # stage1_3_residual = self.stage1[11:14]
        # stage1_3_shortcut = nn.Identity()
        # residual = stage1_3_residual(residual)
        # shortcut = stage1_3_shortcut(shortcut)
        # x = torch.cat([shortcut, residual], dim=1)
        # x = channel_shuffle(x, 2)


        "stage2的7个block"
        # shortcut, residual = channel_split(x,24)
        # x = self.stage1(x)
        shortcut=x; residual=x
        stage2_0_residual = self.stage2[:3]
        stage2_0_shortcut = self.stage2[3:5]
        residual = stage2_0_residual(residual)
        shortcut = stage2_0_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        stage2_channel = 48
        shortcut, residual = channel_split(x,stage2_channel)
        stage2_1_residual = self.stage2[5:8]
        stage2_1_shortcut = nn.Identity()
        residual = stage2_1_residual(residual)
        shortcut = stage2_1_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_2_residual = self.stage2[8:11]
        stage2_2_shortcut = nn.Identity()
        residual = stage2_2_residual(residual)
        shortcut = stage2_2_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_3_residual = self.stage2[11:14]
        stage2_3_shortcut = nn.Identity()
        residual = stage2_3_residual(residual)
        shortcut = stage2_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_4_residual = self.stage2[14:17]
        stage2_4_shortcut = nn.Identity()
        residual = stage2_4_residual(residual)
        shortcut = stage2_4_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_5_residual = self.stage2[17:20]
        stage2_5_shortcut = nn.Identity()
        residual = stage2_5_residual(residual)
        shortcut = stage2_5_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_6_residual = self.stage2[20:23]
        stage2_6_shortcut = nn.Identity()
        residual = stage2_6_residual(residual)
        shortcut = stage2_6_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage2_channel)
        stage2_6_residual = self.stage2[23:26]
        stage2_6_shortcut = nn.Identity()
        residual = stage2_6_residual(residual)
        shortcut = stage2_6_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)





        "stage3的4个block"
        # x = self.stage2(x)
        # shortcut, residual = channel_split(x, 48)
        shortcut = x; residual = x
        stage3_0_residual = self.stage3[:3]
        stage3_0_shortcut = self.stage3[3:5]
        residual = stage3_0_residual(residual)
        shortcut = stage3_0_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        stage3_channel = 96
        shortcut, residual = channel_split(x, stage3_channel)
        stage3_1_residual = self.stage3[5:8]
        stage3_1_shortcut = nn.Identity()
        residual = stage3_1_residual(residual)
        shortcut = stage3_1_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_2_residual = self.stage3[8:11]
        stage3_2_shortcut = nn.Identity()
        residual = stage3_2_residual(residual)
        shortcut = stage3_2_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[11:14]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[14:17]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[17:20]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[20:23]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[23:26]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[26:29]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        shortcut, residual = channel_split(x, stage3_channel)
        stage3_3_residual = self.stage3[29:32]
        stage3_3_shortcut = nn.Identity()
        residual = stage3_3_residual(residual)
        shortcut = stage3_3_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        # x = self.stage3(x)

        "stage4 一个block"
        # shortcut, residual = channel_split(x, 96)
        stage4_channel = 192
        shortcut = x; residual = x
        stage4_0_residual = self.stage4[:3]
        stage4_0_shortcut = self.stage4[3:5]
        residual = stage4_0_residual(residual)
        shortcut = stage4_0_shortcut(shortcut)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)
        shortcut, residual = channel_split(x,stage4_channel)

        # x = self.stage4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (1, 1.0, 1.0, 1),
           "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0),
           "use_se": True},
}


def mobileone(num_classes: int = 1000, inference_mode: bool = False,
              variant: str = "s0") -> nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode,
                     **variant_params)


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model

if __name__ == '__main__':
    import torch

    model = mobileone(variant="s0", inference_mode=True, num_classes=1000)
    print(model)
    random_input = torch.randn(1,3,224,224)
    output = model(random_input)
    print(output.shape)

    # 计算该网络的参数量
    from thop import profile
    from thop import clever_format
    flops, params = profile(model, inputs=(random_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

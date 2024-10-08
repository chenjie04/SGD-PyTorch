from typing import Union, List, Tuple
import math

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.model import constant_init, kaiming_init
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.models.layers.yolo_bricks import CSPLayerWithTwoConv


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


class Stem(BaseModule):
    """Stem module for Canny backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv_cfg: ConfigType = None,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
    ):
        super().__init__()

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialChannelDownsample(BaseModule):
    """patial-channel decoupled downsampling module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        conv_cfg: ConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
    ):
        super().__init__()
        self.channel_expansion = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.spatial_downsample = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=out_channels,
            conv_cfg=conv_cfg,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.spatial_downsample(self.channel_expansion(x))


@MODELS.register_module()
class CandyNetBackbone(BaseBackbone):
    """CannyNet backbone."""

    arch_settings = {
        "P5": [
            # in_channels, out_channels, num_blocks, use_scdown, useself_attention
            [64, 128, 3, False, False],
            [128, 256, 6, True, True],
            [256, 512, 6, True, True],
            [512, 1024, 3, True, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        img_scale: Tuple = (640, 640),
        last_stage_out_channels: int = 1024,
        plugins: Union[dict, List[dict]] = None,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        expansion: float = 0.5,
        num_elans: int = 4,
        out_indices: Tuple[int] = (2, 3, 4),
        frozen_stages: int = -1,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
    ):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.conv_cfg = conv_cfg
        self.expansion = expansion

        self.num_attn = 0
        self.num_elans = num_elans
        # Pytorch中特征图维度顺序为：[batch_size, channel, height, width]

        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg,
        )

        self.init_weights()

    def build_stem_layer(self):
        return Stem(
            in_channels=self.input_channels,
            out_channels=make_divisible(self.arch_setting[0][0], self.widen_factor),
            conv_cfg=self.conv_cfg,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_stage_layer(self, stage_idx: int, setting: list):
        (
            in_channels,
            out_channels,
            num_blocks,
            use_scdown,
            use_attn,
        ) = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []

        if use_scdown:
            sc_down_layer = SpatialChannelDownsample(
                in_channels,
                out_channels,
                conv_cfg=self.conv_cfg,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(sc_down_layer)
        else:
            conv_layer = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(conv_layer)

        if not use_attn:
            blocks = CSPLayerWithTwoConv(
                in_channels=out_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                add_identity=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(blocks)
        else:
            from models.layers.efficient_attention import EfficientAttentionELANBlock
            blocks = nn.Sequential(
                *[
                    EfficientAttentionELANBlock(
                        model_dim=out_channels,
                        key_dim=out_channels, # 下次测试 key_dim = out_channels // 2
                        value_dim=out_channels,
                        num_elans=self.num_elans,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                    for _ in range(num_blocks)
                ]
            )

            
            stage.append(blocks)
            self.num_attn += 1

        return stage

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(
                        m, mode="fan_out", nonlinearity="relu", distribution="normal"
                    )  # leaky_relu
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1, bias=0)
        else:
            super().init_weights()


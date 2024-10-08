import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


# ----------------------------------------------------------------------
class EfficientAttention(BaseModule):
    def __init__(
        self,
        model_dim: int = 512,
        key_dim: int = 64,
        value_dim: int = 512,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.keys = ConvModule(
            model_dim,
            key_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.queries = ConvModule(
            model_dim,
            key_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.values = ConvModule(
            model_dim,
            value_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, P, N] eg. [B, 512, 80, 160]
        B, C, P, N = x.shape
        queries = self.queries(x).reshape(B, -1, P * N)
        keys = self.keys(x).reshape(B, -1, P * N)
        values = self.values(x).reshape(B, -1, P * N)

        # [B, d_k, P * N] @ [B, N * P, d_v] -> [B, d_k, d_v]
        context = F.softmax(keys, dim=-1) @ values.transpose(1, 2).contiguous()
        # [B, P * N, d_k] @ [B, d_k, d_v] -> [B, P * N, d_v]
        attn_values = F.softmax(queries.transpose(1, 2).contiguous(), dim=-1) @ context
        # [B, P * N, d_v] -> [B, d_v, P, N]
        attn_values = attn_values.transpose(1, 2).contiguous().reshape(B, -1, P, N)

        return attn_values


class EfficientAttentionELANBlock(BaseModule):

    def __init__(
        self,
        model_dim: int = 512,
        key_dim: int = 64,
        value_dim: int = 512,
        num_elans: int = 3,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.attn_layer = EfficientAttention(
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.elan_layer = ELANBlock(
            in_channels=value_dim,
            out_channels=value_dim,
            middle_ratio=0.5,
            block_ratio=0.5,
            num_blocks=num_elans,  # 改为1时，性能下降太多
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        if model_dim == value_dim:
            self.shortcut = True
        else:
            self.shortcut = False

    def forward(self, x: torch.Tensor):
        if self.shortcut:
            shortcut = x
            x = self.attn_layer(x) + shortcut
            shortcut = x
            x = self.elan_layer(x) + shortcut
        else:
            x = self.attn_layer(x)
            x = self.elan_layer(x)
        return x


from mmyolo.models.layers.yolo_bricks import DarknetBottleneck


class ELANBlock(BaseModule):
    """Efficient layer aggregation networks for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        middle_ratio (float): The scaling ratio of the middle layer
            based on the in_channels.
        block_ratio (float): The scaling ratio of the block layer
            based on the in_channels.
        num_blocks (int): The number of blocks in the main branch.
            Defaults to 2.
        num_convs_in_block (int): The number of convs pre block.
            Defaults to 1.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        middle_ratio: float,
        block_ratio: float,
        num_blocks: int = 2,
        num_convs_in_block: int = 1,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert num_blocks >= 1
        assert num_convs_in_block >= 1

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * block_ratio)
        final_conv_in_channels = int(num_blocks * block_channels) + 2 * middle_channels

        self.main_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.short_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_convs_in_block == 1:
                # internal_block = ConvModule(
                #     middle_channels,
                #     block_channels,
                #     3,
                #     padding=1,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                # )
                internal_block = DarknetBottleneck(
                    middle_channels,
                    block_channels,
                    expansion=0.5,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            else:
                internal_block = []
                for _ in range(num_convs_in_block):
                    internal_block.append(
                        # ConvModule(
                        #     middle_channels,
                        #     block_channels,
                        #     3,
                        #     padding=1,
                        #     conv_cfg=conv_cfg,
                        #     norm_cfg=norm_cfg,
                        #     act_cfg=act_cfg,
                        # )
                        DarknetBottleneck(
                            middle_channels,
                            block_channels,
                            expansion=0.5,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        )
                    )
                    middle_channels = block_channels
                internal_block = nn.Sequential(*internal_block)

            middle_channels = block_channels
            self.blocks.append(internal_block)

        self.final_conv = ConvModule(
            final_conv_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.final_conv(x_final)

class EfficientAttentionFFN(BaseModule):

    def __init__(
        self,
        model_dim: int = 512,
        key_dim: int = 64,
        value_dim: int = 512,
        num_elans: int = 3,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.attn_layer = EfficientAttention(
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.ffn_layer = nn.Sequential(
            nn.Conv2d(model_dim, model_dim * 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(model_dim * 4, model_dim, kernel_size=1),
        )


    def forward(self, x: torch.Tensor):
        x = self.attn_layer(x)
        x = self.ffn_layer(x)
        return x

class ChannelAttention(nn.Module):
    """Channel Attention Module.

    Args:
        channels (int): Number of channels.
        reduction (int): Reduction ratio.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
    ):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.expansion = 2  # a, b

        hard_sigmoid = dict(type="HSigmoid", bias=3.0, divisor=6.0)

        self.fc = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduction),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg,
            ),
            ConvModule(
                in_channels=int(channels / reduction),
                out_channels=channels * self.expansion,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=hard_sigmoid,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        a_avg, b_avg = torch.split(avg_out, self.channels, dim=1)
        a_max, b_max = torch.split(max_out, self.channels, dim=1)
        # out = a_avg * x + b_avg + a_max * x + b_max
        out = torch.max(a_avg * x + b_avg, a_max * x + b_max)

        return out


class SpatialAttention(nn.Module):
    """SpatialAttention
    Args:
         kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 3.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()

        self.conv = ConvModule(
            in_channels=1,
            out_channels=2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.conv(avg_out)
        max_out = self.conv(max_out)
        a_avg, b_avg = torch.split(avg_out, 1, dim=1)
        a_max, b_max = torch.split(max_out, 1, dim=1)
        # out = a_avg * x + b_avg + a_max * x + b_max
        out = torch.max(a_avg * x + b_avg, a_max * x + b_max)
        return out


class CSAttention(nn.Module):
    """Channel-Spatial Attention Module.

    Args:
        channels (int): Number of channels.
        reduction (int): Reduction ratio.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 3.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 3,
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
    ):
        super().__init__()

        self.layer = nn.Sequential(
            ChannelAttention(channels=channels, reduction=reduction, act_cfg=act_cfg),
            SpatialAttention(kernel_size=kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.layer(x)
        return out

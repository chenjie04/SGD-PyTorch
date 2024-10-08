from typing import List, Tuple
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from models.layers.efficient_attention import EfficientAttentionELANBlock
from models.utils.misc import make_divisible
from mmyolo.registry import MODELS


class SelfAttnFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: list = [256, 512, 1024],
        widen_factor: float = 1.0,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.act_cfg = act_cfg

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self.self_attn_modules = nn.ModuleList(
            EfficientAttentionELANBlock(
                model_dim=self.in_channels[i],
                key_dim=self.in_channels[i],
                value_dim=self.in_channels[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for i in range(len(self.in_channels))
        )

        self.down_sampling = nn.ModuleList(
            ConvModule(
                in_channels=self.in_channels[i],
                out_channels=self.in_channels[i + 1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for i in range(len(self.in_channels) - 1)
        )

        self.up_sampling = nn.ModuleList(
            nn.Sequential(
                nn.PixelShuffle(2),
                ConvModule(
                    in_channels=self.in_channels[i + 1] // 4,
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            for i in range(len(self.in_channels) - 1)
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        inputs: 0 torch.Size([2, 256, 80, 80])
                1 torch.Size([2, 512, 40, 40])
                2 torch.Size([2, 1024, 20, 20])
        """
        outs = []
        idx_down = 0
        idx_up = 0
        for level in range(len(inputs)):
            mid_feats = inputs[level]
            sum_feats = mid_feats
            if level > 0:
                low_feats = self.down_sampling[idx_down](inputs[level - 1])
                idx_down += 1
                sum_feats = sum_feats + low_feats
            if level < len(inputs) - 1:
                high_feats = self.up_sampling[idx_up](inputs[level + 1])
                idx_up += 1
                sum_feats = sum_feats + high_feats

            sum_feats = self.self_attn_modules[level](sum_feats)
            outs.append(sum_feats)
        return outs


class SelfAttnFusionBlock_v1(nn.Module):
    def __init__(
        self,
        in_channels: list = [256, 512, 1024],
        widen_factor: float = 1.0,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.act_cfg = act_cfg

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self.self_attn_modules = nn.ModuleList(
            EfficientAttentionELANBlock(
                model_dim=self.in_channels[i],
                key_dim=self.in_channels[i],
                value_dim=self.in_channels[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for i in range(len(self.in_channels))
        )

        self.down_sampling = nn.ModuleList(
            nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                ConvModule(
                    in_channels=self.in_channels[i + 1],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.in_channels[i + 1],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            for i in range(len(self.in_channels) - 1)
        )

        self.up_sampling = nn.ModuleList(
            nn.Sequential(
                nn.PixelShuffle(2),
                ConvModule(
                    in_channels=self.in_channels[i + 1] // 4,
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            for i in range(len(self.in_channels) - 1)
        )

        self.concat_fusion = nn.ModuleList(
            ConvModule(
                in_channels=self.in_channels[i] * 2,
                out_channels=self.in_channels[i],
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for i in range(len(self.in_channels))
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        inputs: 0 torch.Size([2, 256, 80, 80])
                1 torch.Size([2, 512, 40, 40])
                2 torch.Size([2, 1024, 20, 20])
        """
        outs = []
        idx_down = 0
        idx_up = 0
        for level in range(len(inputs)):
            mid_feats = inputs[level]
            sum_feats = mid_feats
            if level > 0:
                low_feats = self.down_sampling[idx_down](inputs[level - 1])
                idx_down += 1
                sum_feats = self.concat_fusion[level](torch.cat([low_feats, sum_feats], dim=1))
            if level < len(inputs) - 1:
                high_feats = self.up_sampling[idx_up](inputs[level + 1])
                idx_up += 1
                sum_feats = self.concat_fusion[level](torch.cat([high_feats, sum_feats], dim=1))

            sum_feats = self.self_attn_modules[level](sum_feats)
            outs.append(sum_feats)
        return outs

class SelfAttnFusionBlock_v2(nn.Module):
    def __init__(
        self,
        in_channels: list = [256, 512, 1024],
        widen_factor: float = 1.0,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.act_cfg = act_cfg

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        from mmyolo.models.plugins.cbam import CBAM
        self.feat_select = nn.ModuleList(
            CBAM(in_channels=self.in_channels[i],kernel_size=3)
            for i in range(len(self.in_channels))
        )

        self.down_sampling = nn.ModuleList(
            nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                ConvModule(
                    in_channels=self.in_channels[i + 1],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.in_channels[i + 1],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            for i in range(len(self.in_channels) - 1)
        )

        self.up_sampling = nn.ModuleList(
            nn.Sequential(
                nn.PixelShuffle(2),
                ConvModule(
                    in_channels=self.in_channels[i + 1] // 4,
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            for i in range(len(self.in_channels) - 1)
        )

        self.concat_fusion = nn.ModuleList()
        for i in range(len(self.in_channels)):
            channel_factor = 1
            if i > 0:
                channel_factor += 1
            if i < len(self.in_channels) - 1:
                channel_factor += 1
            self.concat_fusion.append(
                ConvModule(
                    in_channels=self.in_channels[i] * channel_factor,
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        from models.layers.efficient_attention import ELANBlock
        self.enhance_modules = nn.ModuleList(
            ELANBlock(in_channels=self.in_channels[i],
                      out_channels=self.in_channels[i],
                      middle_ratio=0.5,
                      block_ratio=0.5,
                      num_blocks=1)
            for i in range(len(self.in_channels))
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        inputs: 0 torch.Size([2, 256, 80, 80])
                1 torch.Size([2, 512, 40, 40])
                2 torch.Size([2, 1024, 20, 20])
        """
        outs = []
        idx_down = 0
        idx_up = 0
        select_feats = []
        for level in range(len(inputs)):
            select_feats.append(self.feat_select[level](inputs[level]))

        for level in range(len(inputs)):
            mid_feats = select_feats[level]
            sum_feats = [mid_feats]
            if level > 0:
                low_feats = self.down_sampling[idx_down](select_feats[level - 1])
                idx_down += 1
                sum_feats.append(low_feats)
            if level < len(inputs) - 1:
                high_feats = self.up_sampling[idx_up](select_feats[level + 1])
                idx_up += 1
                sum_feats.append(high_feats)
            sum_feats = self.concat_fusion[level](torch.cat(sum_feats, dim=1))
            sum_feats = self.enhance_modules[level](sum_feats)
            outs.append(sum_feats)
        return outs

@MODELS.register_module()
class SelfAttentionNeck(nn.Module):
    def __init__(
        self,
        in_channels: list = [256, 512, 1024],
        widen_factor: float = 1.0,
        num_blocks: int = 2,  # The number of SelfAttnFusionBloack
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()

        self.fusion_blocks = nn.Sequential(
            *[
                SelfAttnFusionBlock_v2(
                    in_channels=in_channels,
                    widen_factor=widen_factor,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.fusion_blocks(inputs)

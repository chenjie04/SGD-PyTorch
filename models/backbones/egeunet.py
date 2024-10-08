import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=dim_in,
        )
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r"""From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format="channels_first"),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                dilation=d_list[0],
                groups=group_size + 1,
            ),
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format="channels_first"),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                dilation=d_list[1],
                groups=group_size + 1,
            ),
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format="channels_first"),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                dilation=d_list[2],
                groups=group_size + 1,
            ),
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format="channels_first"),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                dilation=d_list[3],
                groups=group_size + 1,
            ),
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format="channels_first"),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(
            xh, size=[xl.size(2), xl.size(3)], mode="bilinear", align_corners=True
        )
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class group_aggregation_bridge_no_mask(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format="channels_first"),
            nn.Conv2d(
                group_size,
                group_size,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                dilation=d_list[0],
                groups=group_size,
            ),
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format="channels_first"),
            nn.Conv2d(
                group_size,
                group_size,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                dilation=d_list[1],
                groups=group_size,
            ),
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format="channels_first"),
            nn.Conv2d(
                group_size,
                group_size,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                dilation=d_list[2],
                groups=group_size,
            ),
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format="channels_first"),
            nn.Conv2d(
                group_size,
                group_size,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                dilation=d_list[3],
                groups=group_size,
            ),
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format="channels_first"),
            nn.Conv2d(dim_xl * 2, dim_xl, 1),
        )

    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(
            xh, size=[xl.size(2), xl.size(3)], mode="bilinear", align_corners=True
        )
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(
            torch.Tensor(1, c_dim_in, x, y), requires_grad=True
        )
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(
            nn.Conv2d(
                c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in
            ),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1),
        )

        self.params_zx = nn.Parameter(
            torch.Tensor(1, 1, c_dim_in, x), requires_grad=True
        )
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(
            nn.Conv1d(
                c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in
            ),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1),
        )

        self.params_zy = nn.Parameter(
            torch.Tensor(1, 1, c_dim_in, y), requires_grad=True
        )
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(
            nn.Conv1d(
                c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in
            ),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in),
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format="channels_first")

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        # ----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(
            F.interpolate(
                params_xy, size=x1.shape[2:4], mode="bilinear", align_corners=True
            )
        )
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(
                params_zx, size=x2.shape[2:4], mode="bilinear", align_corners=True
            ).squeeze(0)
        ).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(
                params_zy, size=x3.shape[2:4], mode="bilinear", align_corners=True
            ).squeeze(0)
        ).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw----------#
        x4 = self.dw(x4)
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class EGEUNet(nn.Module):

    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        bridge=True,
        gt_ds=True,
    ):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print("group_aggregation_bridge was used")
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print("gt deep supervision was used")

        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(
                gt_pre5, scale_factor=32, mode="bilinear", align_corners=True
            )
        else:
            t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(
            F.interpolate(
                self.dbn2(self.decoder2(out5)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c3, H/16, W/16
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(
                gt_pre4, scale_factor=16, mode="bilinear", align_corners=True
            )
        else:
            t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(
            F.interpolate(
                self.dbn3(self.decoder3(out4)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c2, H/8, W/8
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(
                gt_pre3, scale_factor=8, mode="bilinear", align_corners=True
            )
        else:
            t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(
            F.interpolate(
                self.dbn4(self.decoder4(out3)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c1, H/4, W/4
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(
                gt_pre2, scale_factor=4, mode="bilinear", align_corners=True
            )
        else:
            t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(
            F.interpolate(
                self.dbn5(self.decoder5(out2)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c0, H/2, W/2
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(
                gt_pre1, scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(
            self.final(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )  # b, num_class, H, W

        if self.gt_ds:
            return (
                torch.sigmoid(gt_pre5),
                torch.sigmoid(gt_pre4),
                torch.sigmoid(gt_pre3),
                torch.sigmoid(gt_pre2),
                torch.sigmoid(gt_pre1),
            ), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)


class SimpleEGEUNet(nn.Module):

    def __init__(self, input_channels=256, bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, stride=1, padding=1),
        )

        self.encoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(
                input_channels // 2, input_channels // 4
            ),
        )

        if bridge:
            self.GAB1 = group_aggregation_bridge_no_mask(
                input_channels // 4, input_channels // 2
            )

        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(
                input_channels // 4, input_channels // 2
            ),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(input_channels // 2, input_channels, 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, input_channels // 2)
        self.ebn2 = nn.GroupNorm(4, input_channels // 4)
        self.dbn1 = nn.GroupNorm(4, input_channels // 2)
        self.dbn2 = nn.GroupNorm(4, input_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c/2, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c/4, H/4, W/4

        out2 = F.gelu(
            F.interpolate(
                self.dbn1(self.decoder1(out)),
                size=t1.size()[2:],
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c/2, H/2, W/2
        t2 = self.GAB1(t2, t1)
        # print(t2.shape, out2.shape)
        out2 = torch.add(out2, t2)  # b, cx2, H/4, W/4

        out1 = F.gelu(
            F.interpolate(
                self.dbn2(self.decoder2(out2)),
                size=x.size()[2:],
                mode="bilinear",
                align_corners=True,
            )
        )  # b, c3, H/16, W/16

        return out1


def channel_wise_norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variances.

    Args:
        feat (torch.Tensor): The original feature map with shape
            (N, C, H, W).
    """
    """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3).contiguous()


class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, loss_weight=10.0):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(
            softmax_pred_T * logsoftmax(preds_T.view(-1, W * H) / self.tau)
            - softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / self.tau)
        ) * (self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss


from typing import List
from mmdet.registry import MODELS
from mmcv.cnn import ConvModule
from mmyolo.models.layers import ELANBlock
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.models.utils import multi_apply, unpack_gt_instances

from mmpretrain.registry import MODELS as Pre_MODELS

@MODELS.register_module()
class CyclicDisentanglement(nn.Module):
    def __init__(
        self,
        in_channels=256,
        num_envs=2,
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],
        ),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # bbox_mAP_copypaste: 0.173 0.332 0.154 0.052 0.222 0.313
        # bbox_mAP_copypaste: 0.155 0.303 0.135 0.049 0.204 0.270 三层
        self.domain_invariant_extractor = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU"),
            ),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU"),
            ),
        )

        self.domain_specific_extractor = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU"),
            ),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU"),
            ),
        )

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.domain_classifier = nn.Linear(in_channels, num_envs)
        self.num_envs = num_envs

        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)

        # bbox_mAP_copypaste: 0.114 0.219 0.102 0.034 0.155 0.222
        # self.domain_invariant_extractor = ELANBlock(
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        #     middle_ratio=0.5,
        #     block_ratio=0.25,
        #     num_blocks=4,
        #     num_convs_in_block=1,
        # )
        # self.domain_specific_extractor = ELANBlock(
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        #     middle_ratio=0.5,
        #     block_ratio=0.25,
        #     num_blocks=4,
        #     num_convs_in_block=1,
        # )

        # bbox_mAP_copypaste: 0.148 0.290 0.129 0.044 0.186 0.271
        # self.domain_invariant_extractor = SimpleEGEUNet(input_channels=in_channels)
        # self.domain_specific_extractor = SimpleEGEUNet(input_channels=in_channels)

    def forward(
        self, feats: List[torch.Tensor], batch_data_samples: SampleList
    ) -> List[torch.Tensor]:

        if not self.training:
            domain_invariant_features = []
            for feat in feats:
                Fdi = self.domain_invariant_extractor(feat)
                domain_invariant_features.append(Fdi)
            return domain_invariant_features
        else:
            loss = dict()
            # -------------------------------------------------------------------
            outputs = unpack_gt_instances(batch_data_samples)
            (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
            env_label = [meta["env_label"] for meta in batch_img_metas]
            env_label = F.one_hot(torch.tensor(env_label), num_classes=self.num_envs).float()
            env_label = env_label.to(feats[0].device)
            # print(env_label)

            gt_bboxes = [gt.bboxes for gt in batch_gt_instances]
            rois = bbox2roi(gt_bboxes)
            # print(rois)
        
            # -------------------------------------------------------------------
            feat_loss = 0.0
            env_cls_loss = 0.0
            con_loss = 0.0
            Feats_di = []
            Feats_ds = []
            Feats_i2i = []
            Feats_i2s = []
            levels = len(feats)
            for feat in feats:
                Fdi = self.domain_invariant_extractor(feat)
                Fds = self.domain_specific_extractor(feat)

                Fi2i = self.domain_invariant_extractor(Fdi)
                Fi2s = self.domain_specific_extractor(Fdi)

                Fs2i = self.domain_invariant_extractor(Fds)
                Fs2s = self.domain_specific_extractor(Fds)

                if torch.isnan(Fdi).any():
                    raise ValueError("NaN value detected in domain invariant features")
                if torch.isinf(Fdi).any():
                    raise ValueError("Inf value detected in domain invariant features")

                Feats_di.append(Fdi)
                Feats_ds.append(Fds)
                Feats_i2i.append(Fi2i)
                Feats_i2s.append(Fi2s)

                B, C, H, W = Fdi.shape

                pred_cls = self.global_average_pooling(Fds)
                pred_cls = pred_cls.view(Fds.shape[0], -1)               
                pred_cls = self.domain_classifier(pred_cls)
                pred_cls = torch.softmax(pred_cls, dim=1)
                # print("\n pred_cls: ", pred_cls, env_label)
                env_cls_loss += F.cross_entropy(pred_cls, env_label) / B

                
                Fdi = Fdi.permute(0, 2, 3, 1).reshape(-1, C)
                Fi2i = Fi2i.permute(0, 2, 3, 1).reshape(-1, C)
                Fi2s = Fi2s.permute(0, 2, 3, 1).reshape(-1, C)
                Fds = Fds.permute(0, 2, 3, 1).reshape(-1, C)
                Fs2i = Fs2i.permute(0, 2, 3, 1).reshape(-1, C)
                Fs2s = Fs2s.permute(0, 2, 3, 1).reshape(-1, C)

                # feat_loss += (
                #     max(
                #         0, torch.exp(F.cosine_similarity(Fdi, Fds).mean() - 1.0)
                #     )
                # )

                sim_di_i2i = F.cosine_similarity(Fdi, Fi2i).mean()
                sim_di_i2s = F.cosine_similarity(Fdi, Fi2s).mean()
                # sim_i2i_s2s = F.cosine_similarity(Fi2i, Fs2s).mean()

                sim_ds_s2s = F.cosine_similarity(Fds, Fs2s).mean()
                sim_ds_s2i = F.cosine_similarity(Fds, Fs2i).mean()
                # sim_s2i_s2s = F.cosine_similarity(Fs2i, Fs2s).mean()

                con_loss += -(
                    torch.log(
                        torch.exp(sim_di_i2i)
                        / (torch.exp(sim_di_i2i) + torch.exp(sim_di_i2s))
                    )
                    + torch.log(
                        torch.exp(sim_ds_s2s)
                        / (torch.exp(sim_ds_s2s) + torch.exp(sim_ds_s2i))
                    )
                ) / B 

            # loss["feat_loss"] = feat_loss
            loss["con_loss"] = con_loss / levels
            loss["env_cls_loss"] = env_cls_loss / (levels)

            # -------------------------------------------------------------------
            roi_feats_di = self.bbox_roi_extractor(Feats_di, rois)
            roi_feats_di = self.global_average_pooling(roi_feats_di)
            roi_feats_di = roi_feats_di.view(Feats_di[0].shape[0], -1)

            roi_feats_i2i = self.bbox_roi_extractor(Feats_i2i, rois)
            roi_feats_i2i = self.global_average_pooling(roi_feats_i2i)
            roi_feats_i2i = roi_feats_i2i.view(Feats_i2i[0].shape[0], -1)

            roi_feats_i2s = self.bbox_roi_extractor(Feats_i2s, rois)
            roi_feats_i2s = self.global_average_pooling(roi_feats_i2s)
            roi_feats_i2s = roi_feats_i2s.view(Feats_i2s[0].shape[0], -1)

            sim_di_i2i = F.cosine_similarity(roi_feats_di, roi_feats_i2i).mean()
            sim_di_i2s = F.cosine_similarity(roi_feats_di, roi_feats_i2s).mean()

            instance_loss = -torch.log(torch.exp(sim_di_i2i) / (torch.exp(sim_di_i2i) + torch.exp(sim_di_i2s)))

            # print("\n instance_loss: ", instance_loss)
            loss["instance_loss"] = instance_loss

            return Feats_di, loss


if __name__ == "__main__":
    x = list()
    x.append(torch.randn(4, 256, 192, 336).cuda())
    x.append(torch.randn(4, 256, 96, 168).cuda())
    x.append(torch.randn(4, 256, 48, 84).cuda())
    x.append(torch.randn(4, 256, 24, 42).cuda())

    feat, loss = CyclicDisentanglement(in_channels=256).cuda()(x)
    print(feat[0].shape, loss)

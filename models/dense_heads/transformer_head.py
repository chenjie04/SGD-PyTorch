from typing import Tuple
import copy
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from mmdet.structures import SampleList
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule


@MODELS.register_module()
class TransformerHead(BaseModule):
    """Head based on transformer for object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
    """

    def __init__(
        self,
        num_classes=80,
        in_channels=1024,
        img_size=640,
        strides=[8, 16, 32],
        num_attenion_blocks=1,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="ReLU", inplace=True),
        one2one_head=dict(
            type="DETRHead",
            num_classes=80,
            embed_dims=1024,
            loss_cls=dict(
                type="CrossEntropyLoss",
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner",
                    match_costs=[
                        dict(type="ClassificationCost", weight=1.0),
                        dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                        dict(type="IoUCost", iou_mode="giou", weight=2.0),
                    ],
                )
            ),
            test_cfg=dict(max_per_img=100),
        ),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.strides = strides

        self.downsampling = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
        )
        self.upsampling = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        self.attention_fusion = AttentionFusionLayer(d_model=in_channels, nhead=32)
        self.num_attenion_blocks = num_attenion_blocks
        self.attention_blocks = nn.ModuleList(
            copy.deepcopy(SelfAttentionLayer(d_model=in_channels, nhead=32))
            for _ in range(self.num_attenion_blocks)
        )

        self.one2one_head = MODELS.build(one2one_head)

    def _forward(self, feats: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (Tensor): Input feature map.
                        [[4, 512, 92, 160],
                        [4, 1024, 46, 80],
                        [4, 2048, 23, 40]]

        Returns:
            Tensor: Detection results.
        """
        # for feat in feats:
        #     print(feat.shape)
        
        value = self.downsampling(feats[0])
        key = feats[1]
        query = self.upsampling(feats[2])

        B, C, H, W = key.shape
        value = value.permute(0, 2, 3, 1).view(B, -1, C)
        key = key.permute(0, 2, 3, 1).view(B, -1, C)
        query = query.permute(0, 2, 3, 1).view(B, -1, C)

        atten_out = self.attention_fusion(query, key, value)
        for attention_block in self.attention_blocks:
            atten_out = attention_block(atten_out)

        return atten_out

    def forward(self, feats: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (Tensor): Input feature map.

        Returns:
            Tensor: Detection results.
        """
        hidden_states = self._forward(feats)
        hidden_states = hidden_states.unsqueeze(0)
        layers_cls_scores, layers_bbox_preds = self.one2one_head(hidden_states)
        print(layers_cls_scores.shape, layers_bbox_preds.shape)
        return layers_cls_scores, layers_bbox_preds

    def loss(self, feats: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        hidden_states = self._forward(feats)
        hidden_states = hidden_states.unsqueeze(0)
        losses = self.one2one_head.loss(hidden_states, batch_data_samples)
        return losses

    def predict(self, feats: Tensor, batch_data_samples: SampleList, rescale: bool = True):
        hidden_states = self._forward(feats)
        hidden_states = hidden_states.unsqueeze(0)
        return self.one2one_head.predict(hidden_states, batch_data_samples, rescale)


class AttentionFusionLayer(BaseModule):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        B, N, C = q.shape
        q, k, v = [
            x.view(B, N, self.nhead, C // self.nhead).transpose(1, 2) for x in (q, k, v)
        ]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.norm(out)
        shortcut = out
        out = self.ffn(out) + shortcut
        return out


class SelfAttentionLayer(BaseModule):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        B, N, C = x.shape
        q, k, v = [
            lin(x).view(B, N, self.nhead, C // self.nhead).transpose(1, 2)
            for lin in self.linears
        ]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.norm1(out) + shortcut
        shortcut = out
        out = self.norm2(self.ffn(out)) + shortcut
        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


if __name__ == "__main__":
    x = list()

    x.append(torch.randn(4, 256, 80, 80).cuda())
    x.append(torch.randn(4, 256, 40, 40).cuda())
    x.append(torch.randn(4, 256, 20, 20).cuda())

    head = TransformerHead().cuda()

    y = head(x)

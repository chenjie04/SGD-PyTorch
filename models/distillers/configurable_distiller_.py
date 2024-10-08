# Copyright (c) OpenMMLab. All rights reserved.

import warnings
import copy

# from inspect import signature
from typing import Dict, List, Optional, Union, Tuple

from mmengine.model import BaseModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from models.distill_algorithms.base import LossResults
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.structures.bbox import bbox_overlaps
from models.task_modules.recorder.recoder_manager import RecorderManager

from models.distillers.base_distiller import BaseDistiller



@MODELS.register_module()
class PKDConfigurableDistiller(BaseDistiller):
    """ConfigurableDistillerInLogits."""

    def __init__(
        self,
        student_recorders: Optional[Dict[str, Dict]] = None,
        teacher_recorders: Optional[Dict[str, Dict]] = None,
        loss_forward_mappings: Optional[Dict[str, Dict]] = None,
        resize_stu: bool = False,
        resize_tea: bool = False,
        loss_weight: float = 20.0,
    ):
        super().__init__()
        # The recorder manager is just constructed, but not really initialized
        # yet. Recorder manager initialization needs to input the corresponding
        # model.
        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)

        if loss_forward_mappings:
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

        self.resize_stu = resize_stu
        self.resize_tea = resize_tea
        self.loss_weight = loss_weight
            

    def prepare_from_student(self, model: BaseModel) -> None:
        """Initialize student recorders."""
        self.student_recorders.initialize(model)

    def prepare_from_teacher(self, model: nn.Module) -> None:
        """Initialize teacher recorders."""
        self.teacher_recorders.initialize(model)

    def get_record(
        self,
        recorder: str,
        from_student: bool,
        record_idx: int = 0,
        data_idx: Optional[int] = None,
        connector: Optional[str] = None,
        connector_idx: Optional[int] = None,
    ) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data

    def layer_norm(self, feat: torch.Tensor) -> torch.Tensor:
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
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def compute_distill_losses(
        self, data_samples: Optional[List] = None
    ) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.

        # 获取教师的特征图
        mapping_teacher = self.loss_forward_mappings["loss_prject_optim"]["preds_T"]
        from_student, recorder, data_idx = (
            mapping_teacher["from_student"],
            mapping_teacher["recorder"],
            mapping_teacher["data_idx"],
        )
        feats_teacher = []
        for idx in data_idx:
            feats_teacher.append(self.get_record(recorder, from_student, data_idx=idx))

        # 获取学生的特征图
        mapping_student = self.loss_forward_mappings["loss_prject_optim"]["preds_S"]
        from_student, recorder, data_idx = (
            mapping_student["from_student"],
            mapping_student["recorder"],
            mapping_student["data_idx"],
        )
        feats_student = []
        for idx in data_idx:
            feats_student.append(self.get_record(recorder, from_student, data_idx=idx))
       

        # 计算损失
        losses = dict()
        feat_loss = 0.0
        for feat_s, feat_t in zip(feats_student, feats_teacher):
            normed_feat_s = self.layer_norm(feat_s)
            normed_feat_t = self.layer_norm(feat_t)
            feat_loss += F.mse_loss(normed_feat_s, normed_feat_t)
       
        losses["feat_loss"] = feat_loss * self.loss_weight
        # ------------------------------------------------------------

        return losses


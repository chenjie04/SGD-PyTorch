# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union, Tuple
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine import MessageHub
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from mmengine.optim import OptimWrapperDict, OptimWrapper
from mmengine.utils import is_list_of
from mmengine import Config

from models.utils.misc import add_prefix
from mmyolo.registry import MODELS
from models.distill_algorithms.base import BaseAlgorithm, LossResults

# torch.autograd.set_detect_anomaly(True)

@MODELS.register_module()
class SingleTeacherDistill_(BaseAlgorithm):
    """``SingleTeacherDistill`` can be used to develop distill algorithms which
    only use one teacher.

    Args:
        distiller (dict): The config dict for built distiller.
        teacher (dict | BaseModel): The config dict for teacher model or built
            teacher model.
        teacher_ckpt (str): The path of teacher's checkpoint. Defaults to None.
        teacher_trainable (bool): Whether the teacher is trainable. Defaults
            to False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to True.
        student_trainable (bool): Whether the student is trainable. Defaults
            to True.
        calculate_student_loss (bool): Whether to calculate student loss
            (original task loss) to update student model. Defaults to True.
        teacher_module_inplace(bool): Whether to allow teacher module inplace
            attribute True. Defaults to False.
    """

    def __init__(
        self,
        distiller: dict,
        teacher: Union[BaseModel, Dict],
        teacher_ckpt: Optional[str] = None,
        teacher_trainable: bool = False,
        teacher_norm_eval: bool = True,
        student_trainable: bool = True,
        calculate_student_loss: bool = True,
        teacher_module_inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.distiller = MODELS.build(distiller)

        if isinstance(teacher, Dict):
            teacher_cfg_path = teacher['cfg_path']
            teacher_cfg = Config.fromfile(filename=teacher_cfg_path)['model']
            teacher = MODELS.build(teacher_cfg)

        if not isinstance(teacher, BaseModel):
            raise TypeError(
                "teacher should be a `dict` or "
                f"`BaseModel` instance, but got "
                f"{type(teacher)}"
            )

        self.teacher = teacher

        # Find all nn.Modules in the model that contain the 'inplace' attribute
        # and set them to False.
        self.teacher_module_inplace = teacher_module_inplace
        if not self.teacher_module_inplace:
            self.set_module_inplace_false(teacher, "self.teacher")

        if teacher_ckpt:
            _ = load_checkpoint(self.teacher, teacher_ckpt)
            # avoid loaded parameters be overwritten
            self.teacher._is_init = True
        self.teacher_trainable = teacher_trainable
        if not self.teacher_trainable:
            for param in self.teacher.parameters():
                param.requires_grad = False
        self.teacher_norm_eval = teacher_norm_eval

        # The student model will not calculate gradients and update parameters
        # in some pretraining process.
        self.student_trainable = student_trainable

        # The student loss will not be updated into ``losses`` in some
        # pretraining process.
        self.calculate_student_loss = calculate_student_loss

        # In ``ConfigurableDistller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        self.distiller.prepare_from_teacher(self.teacher)

        # may be modified by stop distillation hook
        self.distillation_stopped = False

        

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        
        if self.teacher_trainable:
            with self.distiller.teacher_recorders:
                teacher_losses = self.teacher(batch_inputs, data_samples, mode="loss")

            losses.update(add_prefix(teacher_losses, "teacher"))
        else:
            with self.distiller.teacher_recorders:
                with torch.no_grad():
                    _ = self.teacher(batch_inputs, data_samples, mode="loss")

        
        if self.calculate_student_loss:
            with self.distiller.student_recorders:
                student_losses = self.student(batch_inputs, data_samples, mode="loss")
            losses.update(add_prefix(student_losses, "student"))
        else:
            with self.distiller.student_recorders:
                if self.student_trainable:
                    _ = self.student(batch_inputs, data_samples, mode="loss")
                else:
                    with torch.no_grad():
                        _ = self.student(batch_inputs, data_samples, mode="loss")

        if not self.distillation_stopped:
            # Automatically compute distill losses based on
            # `loss_forward_mappings`.
            # The required data already exists in the recorders.
            distill_losses = self.distiller.compute_distill_losses(data_samples)
            losses.update(add_prefix(distill_losses, "distill"))

        return losses

    def train(self, mode: bool = True) -> None:
        """Set distiller's forward mode."""
        super().train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # loss = sum(value for key, value in log_vars if 'loss' in key and 'distill.teacher' not in key)
        loss = sum(value for key, value in log_vars if 'loss' in key) #   and 'distill.student' not in key
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore


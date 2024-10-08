# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import copy
import torch
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNN_(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 feat_refiner: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        
        self.feat_refiner = MODELS.build(feat_refiner)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.training:
            x = self.backbone(batch_inputs)
            x, feat_loss = self.feat_refiner(x)
            if self.with_neck:
                x = self.neck(x)
            return x, feat_loss
        else:
            x = self.backbone(batch_inputs)
            x = self.feat_refiner(x)
            if self.with_neck:
                x = self.neck(x)
            return x
    
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        if self.training:
            results = ()
            x, feat_loss = self.extract_feat(batch_inputs)

            if self.with_rpn:
                rpn_results_list = self.rpn_head.predict(
                    x, batch_data_samples, rescale=False)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            roi_outs = self.roi_head.forward(x, rpn_results_list,
                                            batch_data_samples)
            results = results + (roi_outs, )
            return results, feat_loss
        else:
            results = ()
            x = self.extract_feat(batch_inputs)

            if self.with_rpn:
                rpn_results_list = self.rpn_head.predict(
                    x, batch_data_samples, rescale=False)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            roi_outs = self.roi_head.forward(x, rpn_results_list,
                                            batch_data_samples)
            results = results + (roi_outs, )
            return results
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x, feat_loss = self.extract_feat(batch_inputs)

        losses = dict()
        losses['feat_loss'] = feat_loss

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

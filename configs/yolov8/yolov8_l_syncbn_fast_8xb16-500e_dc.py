# ==============================
# Input shape: torch.Size([640, 640])
# Model Flops: 82.568G
# Model Parameters: 43.635M
# ==============================
_base_ = './yolov8_m_syncbn_fast_8xb16-500e_dc.py'

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

save_epoch_intervals = 10

# Batch size of a single GPU during training
train_batch_size_per_gpu = 2

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
mosaic_affine_transform = _base_.mosaic_affine_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth",
            prefix="backbone",
        ),),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth",
            prefix="neck",
        ),),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])),
            init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth",
            prefix="bbox_head",
        ),)

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

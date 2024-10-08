# ==============================
# Input shape: torch.Size([640, 640])
# Model Flops: 0.126T
# Model Parameters: 57.753M
# ==============================
_base_ = "./yolov8_m_syncbn_fast_8xb16-500e_dc.py"

custom_imports = dict(
    imports=[
        "models.backbones.candy_backbone",
    ],
    allow_failed_imports=False,
)

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
weight_decay = 0.0005

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
        type="CandyNetBackbone",
        arch="P5",
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_elans=4,
        expansion=0.5,
        out_indices=(2, 3, 4),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
    ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor, in_channels=[256, 512, last_stage_out_channels]
        )
    ),
)

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type="YOLOv5MixUp",
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform],
    ),
    *last_transform,
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu,dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(
    type="AmpOptimWrapper",
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type="SGD",
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu,
    ),
    constructor="YOLOv5OptimizerConstructor",
)

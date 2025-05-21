_base_ = './yolov8_m_cityscape.py'

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

train_batch_size_per_gpu = 4

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
mosaic_affine_transform = _base_.mosaic_affine_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])))

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu, dataset=dict(pipeline=train_pipeline))
base_lr = 0.001
weight_decay = 0.0005
optim_wrapper = dict(
    # type='OptimWrapper',
    type='AmpOptimWrapper',
    # dtype='float16',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

load_from = './checkpoints/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'  
# Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False 

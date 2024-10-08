# ==============================
# Input shape: torch.Size([640, 640])
# Model Flops: 4.073G
# Model Parameters: 3.012M
# ==============================

_base_ = "./yolov8_s_syncbn_fast_8xb16-500e_voc.py"

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="checkpoints/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth",
        #     prefix="backbone",
        # ),
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="checkpoints/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth",
        #     prefix="neck",
        # ),
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor),
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="checkpoints/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth",
        #     prefix="bbox_head",
        # ),
    ),
)

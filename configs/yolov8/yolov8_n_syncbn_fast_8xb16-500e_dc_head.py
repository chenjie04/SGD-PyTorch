_base_ = "./yolov8_s_syncbn_fast_8xb16-500e_dc.py"

custom_imports = dict(
    imports=[
        "models.dense_heads.cbam_head",
    ],
    allow_failed_imports=False,
)

deepen_factor = 0.33
widen_factor = 0.25

num_classes = 7
strides = [8, 16, 32]

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(
            type="SEHeadModule",
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
            act_cfg=dict(type="SiLU", inplace=True),
            featmap_strides=strides,
        ),
    ),
)

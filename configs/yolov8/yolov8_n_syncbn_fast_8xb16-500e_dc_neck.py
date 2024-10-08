_base_ = "./yolov8_s_syncbn_fast_8xb16-500e_dc.py"

randomness=dict(seed=241353730)

custom_imports = dict(
    imports=[
        "models.necks.attn_pafpn",
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
        type="PAFPNWithAttn",
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=False),
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor),
    ),
)

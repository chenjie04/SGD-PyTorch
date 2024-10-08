# ==============================
# Input shape: torch.Size([640, 640])
# Model Flops: 4.724G
# Model Parameters: 2.982M
# ==============================
_base_ = "./yolov8_s_syncbn_fast_8xb16-500e_voc.py"

custom_imports = dict(
    imports=[
        "models.backbones.candy_backbone_attn_ffn",
    ],
    allow_failed_imports=False,
)

deepen_factor = 0.33
widen_factor = 0.25

img_scale = (640, 640)

model = dict(
    backbone=dict(
        type="CandyNetBackboneAttnFFN",
        arch="P5",
        img_scale=img_scale,
        last_stage_out_channels=1024,
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
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor),
    ),
)

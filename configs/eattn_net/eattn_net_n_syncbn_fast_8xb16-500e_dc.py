# ==============================
# Input shape: torch.Size([640, 640])
# Model Flops: 5.189G
# Model Parameters: 3.234M
# ==============================
_base_ = './eattn_net_s_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
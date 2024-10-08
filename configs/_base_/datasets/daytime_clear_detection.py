# dataset settings
dataset_type = "CocoDataset"
data_root = "data/Diverse_Weather_Dataset/daytime_clear/"
# env_class_names = ['daytime_clear', 'daytime_foggy', 'dusk_rainy', 'night_rainy', 'Night-Sunny'] 
test_data_root = "data/Diverse_Weather_Dataset/daytime_foggy/"

classes = ("bus", "bike", "car", "motor", "person", "rider", "truck")
metainfo = dict(classes=classes)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None
img_scale = (1280, 800)  # width, height 原图大小（1280, 720）
# img_scale = (1333, 800)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "env_label",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# data_root = 'data/Diverse_Weather_Dataset/daytime_clear/'

daytime_clear_dataset = dict(
    type='CocoDataset_with_env',
    data_root="data/Diverse_Weather_Dataset/daytime_clear/",
    metainfo=metainfo,
    ann_file="voc07_train.json",
    data_prefix=dict(img=""),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)

daytime_foggy_dataset = dict(
    type='CocoDataset_with_env',
    data_root="data/Diverse_Weather_Dataset/daytime_foggy/",
    metainfo=metainfo,
    ann_file="voc07_train.json",
    data_prefix=dict(img=""),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)

Night_Sunny_dataset = dict(
    type='CocoDataset_with_env',
    data_root="data/Diverse_Weather_Dataset/Night-Sunny/",
    metainfo=metainfo,
    ann_file="voc07_train.json",
    data_prefix=dict(img=""),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)

night_rainy_dataset = dict(
    type='CocoDataset_with_env',
    data_root="data/Diverse_Weather_Dataset/night_rainy/",
    metainfo=metainfo,
    ann_file="voc07_train.json",
    data_prefix=dict(img=""),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)

concat_dataset = dict(
    type="ConcatDataset",
    datasets=[daytime_clear_dataset, daytime_foggy_dataset],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=daytime_clear_dataset,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="voc07_train.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# test_dataloader = val_dataloader

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=test_data_root,
        metainfo=metainfo,
        ann_file="voc07_train.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "voc07_train.json",
    metric="bbox",
    format_only=False,
    classwise=True,
    backend_args=backend_args,
)

# test_evaluator = val_evaluator

test_evaluator = dict(
    type="CocoMetric",
    ann_file=test_data_root + "voc07_train.json",
    metric="bbox",
    format_only=False,
    classwise=True,
    backend_args=backend_args,
)

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')

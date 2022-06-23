_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True

distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='Feat_L2',
                                       name='loss_fgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='Feat_L2',
                                       name='loss_fgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='Feat_L2',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='Feat_L2',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='Feat_L2',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_r101_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)

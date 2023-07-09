model = dict(
    type='OneMap',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v1',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='OneMap_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=1,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        # loss_kernel=dict(
        #     type='DiceLoss',
        #     loss_weight=0.5
        # ),
        # loss_emb=dict(
        #     type='EmbLoss_v1',
        #     feature_dim=4,
        #     loss_weight=0.25
        # )
    )
)
data = dict(
    batch_size=2,
    train=dict(
        type='PAN_IC15_Container_1class',
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.5,
        read_type='cv2',
        # pretrain='checkpoints/pan_r18_ic15_container/checkpoint_600ep.pth.tar'
    ),
    test=dict(
        type='PAN_IC15_Container_1class',
        split='test',
        short_size=736,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=300,
    optimizer='SGD'
)

test_cfg = dict(
    min_score=0.85,
    min_area=16,
    bbox_type='rect',
    result_path='outputs/submit_ic15_container_1class.zip'
)


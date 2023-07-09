model = dict(
    type='TwoMap',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='TwoMap_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=2,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        # use_coordconv=False,
        # loss_kernel=dict(
        #     type='DiceLoss',
        #     loss_weight=0.5
        # ),
        # loss_emb=dict(
        #     type='EmbLoss_v1',
        #     feature_dim=4,
        #     loss_weight=0.25
        # )
    ),
    recognition_head=dict(
        type='PAN_PP_RecHead',
        input_dim=512,
        hidden_dim=128,
        feature_size=(12, 72)
    )
)
data = dict(
    batch_size=4,
    train=dict(
        type='PAN_PP_IC15_Container_2classes',
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.5,
        read_type='cv2',
        with_rec=True
    ),
    test=dict(
        type='PAN_PP_IC15_Container_2classes',
        split='test',
        short_size=736,
        read_type='cv2',
        with_rec=True
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=300,
    optimizer='SGD',
    use_ex=False,
    pretrain='/data/ys/PycharmProjects2/pan_pp.pytorch/checkpoints/threemap_r18_ic15_container/checkpoint_180ep.pth.tar'
)

test_cfg = dict(
    min_score=0.8,
    min_area=260,
    min_kernel_area=2.6,
    scale=4,
    bbox_type='rect',
    result_path='outputs/submit_con_rec.zip',
    rec_post_process=dict(
        len_thres=3,
        score_thres=0.95,
        unalpha_score_thres=0.9,
        ignore_score_thres=0.93,
        editDist_thres=2,
        voc_path=None  # './data/ICDAR2015/Challenge4/GenericVocabulary.txt'
    ),
)

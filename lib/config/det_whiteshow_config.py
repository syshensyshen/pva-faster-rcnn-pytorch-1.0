model_cfg = dict(
    class_list=(["__background__", 'other', 'scratch_deep', 'scratch_shallow', 'fracture']),
    defect_class_list=(["SL", "YH"]),
    rename_class_list={},
    confidence_per_cls={'SL': [0.1], 'YH': [0.1], 'FC': [
        0.98], 'QZ': [0.99], 'LH': [0.39], 'AJ': [0.23]},
    ANCHOR_SCALES=[1.0, 2.5, 6, 10, 14, 20, 40, 80, 120],
    ANCHOR_RATIOS=[0.0667, 0.133, 0.333, 0.5,
                   0.667, 1.0, 1.5, 2.0, 3.0, 7.5, 15],
    POOLING_MODE='align',
    POOLING_SIZE=7,
    FEAT_STRIDE=[4],

    image_postfix=".jpg",
    model=None,
    save_model_interval=5,

    MODEL=dict(
        RCNN_CIN=1024,
        RPN_CIN=256,
        RCNN_LAST=1024,
        BACKBONE='shortshortlitehyper',
        DOUT_BASE_MODEL=176,
    ),

    TRAIN=dict(
        dataset="xml",
        train_path="/data/ssy/outwrad-1024/train",  # hard_sample
        val_path=None,
        save_dir="save_models/",
        # resume_model=None,
        resume_model=None,
        model_name="outward_",
        gpus=(0, ),
        batch_size_per_gpu=4,
        epochs=1000,
        num_works=2,
        loss_type="smoothL1loss",  # "smoothL1loss" "GIOUloss" "IOUloss"
        is_ohem_rpn=0,
        is_ohem_rcnn=0,
        LEARNING_RATE=0.001,
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0005,
        BATCH_SIZE=128,  # rcnn
        FG_FRACTION=0.25,  # rcnn
        FG_THRESH=0.5,  # rcnn
        BG_THRESH_HI=0.5,  # rcnn
        BG_THRESH_LO=0.0,  # rcnn
        RPN_POSITIVE_OVERLAP=0.7,
        RPN_NEGATIVE_OVERLAP=0.3,
        RPN_FG_FRACTION=0.5,
        RPN_BATCHSIZE=256,
        RPN_NMS_THRESH=0.7,
        RPN_PRE_NMS_TOP_N=12000,
        RPN_POST_NMS_TOP_N=2000,
        RPN_BBOX_INSIDE_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
        TRUNCATED=False,
        BBOX_NORMALIZE_MEANS=(0.0, 0.0, 0.0, 0.0),
        BBOX_NORMALIZE_STDS=(0.1, 0.1, 0.2, 0.2),
        BBOX_INSIDE_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
        RPN_MIN_SIZE=4,
        RPN_CLOBBER_POSITIVES=False,
        RPN_POSITIVE_WEIGHT=-1.0,
        BBOX_NORMALIZE_TARGETS_PRECOMPUTED=True,
    ),

    TEST=dict(
        img_path="/data0/datasets/det_whiteshow_defect/1/hard_sample",
        save_dir="/data/zhangcc/data/tmp/test_results/det_whiteshow_defect",
        gpus=(0, ),
        SCALES=(600,),
        MAX_SIZE=2880,
        TEST_SCALE=1024,
        SCALE_MULTIPLE_OF=32,
        iou_thresh=0.5,
        nms_thresh=0.25,
        thresh=0.5,
        small_object_size=0,
        RPN_PRE_NMS_TOP_N=6000,
        RPN_POST_NMS_TOP_N=512,  # 300,
        RPN_NMS_THRESH=0.7,
        RPN_MIN_SIZE=4,
        gama=False,
    ),
)

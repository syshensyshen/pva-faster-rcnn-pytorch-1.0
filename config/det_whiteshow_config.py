gpus=['1',]
dataset="xml"
train_path="/data/ssy/safe--0811/train"
val_path=None
save_dir="save_models/"
resume_model=''
model_name='safe_0811_'
lr=0.001
epochs=100
batch_size_per_gpu=6
num_works=6
class_list=(['hook', 'slide block', 'together'])
defect_class_list=(['hook', 'slide block', 'together'])
rename_class_list={},
confidence_per_cls={'hook': [0.1], 'slide block': [0.1], 'together': [0.98]}
save_model_interval=10
models = dict(
    structs = {'backbone': 'resnet', 'featurescompose': 'hypernet', 'rpn_tools': 'rpn_tools', 'rcnn_tools': 'rcnn_tools'},
    backbone=dict(
        depth=50,
        pretrained=True,
    ),
    featurescompose=dict(
        hyper_dim=2,
        inchannels=[256, 512, 1024, 2048],
        output_channels=256,
    ),
    rpn_tools=dict(
        stride=16,
        scales=[8, 16, 32],
        ratios=[0.5, 1.0, 2.0],
        inchannels=256,
        output_channels = 256,
        pos_thresh=0.5,
        neg_thresh=0.5,
        # min_pos_iou=0.5,
        fraction = 0.25,
        # total number of examples
        batch_size = 256,
        # nms threshold used on rpn proposals
        nms_thresh = 0.7,
        # number of top scoring boxes to keep before apply nms to rpn proposals
        pre_nms = 12000,
        # number of top scoring boxes to keep after applying nms to rpn proposals
        post_nms = 6000,
        # proposal height and width both need to be greater than rpn_min_size (at orig image scale)
        min_size = 8,
        # deprecated (outside weights)
        inside_weight = 1.0,
        clobber_positive=False,
        # give the positive rpn examples weight of p * 1 / {num positives}
        # and give negatives a weight of (1 - p)
        # set to -1.0 to use uniform example weighting
        pos_weight = -1.0,
    ),
    rcnn_tools=dict(
        ohem_rcnn=False,
        class_list=class_list,
        stride = 16,
        inchannels = 256,
        pooling_size = 7,
        rcnn_first = 1024,
        rcnn_last_din = 1024,
        mode = 'align',
        mean = [0,0,0,0],
        std = [0.1,0.1,0.2,0.2],
        inside_weight = [1.0,1.0,1.0,1.0],
        fraction = 0.25,
        batch_size = 128,
        fg_thresh = 0.5,
        bg_thresh_hi = 0.5,
        bg_thresh_lo = 0.5,
        with_avg_pool = False,
        class_agnostic = False,
    )
)

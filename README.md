functions:
pva-pytorch using pytorch-1.0
add multi-scales image train

`pytorch1.0`,`opencv-python`,`skimage`


CUDA_VISIBLE_DEVICES=1 python trainval_net.py --batch_size 8 \
       --save_dir ./save_models \
       --xml_path /data/datasets/defect_detection/voc/VOC2007/Annotations/ \
       --img_path /data/datasets/defect_detection/voc/VOC2007/JPEGImages/ \
       --network pva --classes 6


pretrainedmode：链接：https://pan.baidu.com/s/1eLdUheV3FHmsyYs-MQLKBQ 提取码：psnw 

 
reference code: 
[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git)
[sanghoon/pytorch_imagenet](https://github.com/sanghoon/pytorch_imagenet.git)



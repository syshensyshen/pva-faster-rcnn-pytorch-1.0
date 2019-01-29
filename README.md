functions:
pva-pytorch using pytorch-1.0
add multi-scales image train

`pytorch1.0`,`opencv-python`,`skimage`

step1: install
git clone https://github.com/syshensyshen/pva-pytorch.git

'cd pva-pytorch/lib'
python setup.py build develop

step2: train
 CUDA_VISIBLE_DEVICES=1 python trainval_net.py --batch_size 8 \
	--save_dir ./save_models \
	--xml_path /data/datasets/defect_detection/voc/VOC2007/Annotations/ \
 	--img_path /data/datasets/defect_detection/voc/VOC2007/JPEGImages/
 
reference code:
 
[faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git)



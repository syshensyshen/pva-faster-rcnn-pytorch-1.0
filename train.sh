CUDA_VISIBLE_DEVICES=1 python trainval_net.py --batch_size 8 \
	--save_dir ./save_models \
	--xml_path /data/datasets/defect_detection/voc/VOC2007/Annotations/ \
	--img_path /data/datasets/defect_detection/voc/VOC2007/JPEGImages/ \
	--network pva --classes 6

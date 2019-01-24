CUDA_VISIBLE_DEVICES=1 python trainval_net.py --batch_size 3 \
	--save_dir ./save_models
	--xml_path /data/ssy/front_parts/VOC2007/Annotations/ \
       --img_path /data/ssy/front_parts/VOC2007/JPEGImages/

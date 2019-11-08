CUDA_VISIBLE_DEVICES=0 python trainval_net.py --batch_size 8 --sub_batch 1 \
	--save_dir ./save_models \
	--xml_path /data/datasets/middle-white-SL-0416/ \
	--img_path /data/datasets/middle-white-SL-0416/ \
	--network lite --classes 5

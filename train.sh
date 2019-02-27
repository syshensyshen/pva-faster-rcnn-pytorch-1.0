CUDA_VISIBLE_DEVICES=1 python trainval_net.py --batch_size 4 \
	--save_dir ./save_models \
	--xml_path /data/ssy/hh-ssy/ \
	--img_path /data/ssy/hh-ssy/ \
	--network resnet_pva --classes 9

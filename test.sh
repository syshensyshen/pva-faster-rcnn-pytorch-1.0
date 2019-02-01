CUDA_VISIBLE_DEVICES=0 python predictor.py \
	--img_path /data/ali-oss/middle-imgs/ \
	--save_dir ./outputs/ \
	--network pva --classes 6 \
	--model ./save_models/phone__039.ckpt

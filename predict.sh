CUDA_VISIBLE_DEVICES=1 python predictor.py \
	--img_path  /data/datasets/middle-white-SL-0416/original/ \
	--save_dir ./outputs/ \
	--network lite --classes 5 \
	--model ./save_models/ws_0416_0002.ckpt

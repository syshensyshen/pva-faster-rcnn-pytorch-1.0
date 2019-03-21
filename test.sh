CUDA_VISIBLE_DEVICES=0 python predictor.py \
	--img_path /data/datasets/defects-checked/20190219-seg_608/ \
	--save_dir ./outputs/ \
	--network lite --classes 2 \
	--model ./save_models/hh_0330.ckpt

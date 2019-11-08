CUDA_VISIBLE_DEVICES=1 python test_net.py \
	--img_path  /data/datasets/defects/middle-bright-HH/ \
	--xml_path /data/datasets/defects/middle-bright-HH/ \
	--save_dir ./outputs/ \
	--network lite --classes 3 \
	--model ./save_models/hh_0325_0005.ckpt

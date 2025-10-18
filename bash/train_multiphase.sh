export CUDA_VISIBLE_DEVICES=0

#cd ..
python  /data/3DUNET_Github/multi_phase/train.py \
--n_gpu \
3 \
--root_path \
/path to your train data  \
--test_path \
/path to your test data \
--module \
net.XXX \
--dataset \
Multiphase \
--crop_inference \
standard_crop_inference \
--eval_interval \
10 \
--max_epochs \
450 \
--batch_size \
12 \
--model_name \
Fusion \
--img_size \
512 \
--base_lr \
0.00011 \
--tag \


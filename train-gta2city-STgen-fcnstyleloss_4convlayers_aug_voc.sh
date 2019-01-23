nvidia-docker run -it \
  -v /mnt/ngv/askc-home/SensorTransfer_datasets/cityscapes-voc-jpg-lab/VOC2012/JPEGImages:/root/dataset_real/VOC2012/JPEGImages:ro \
  -v /mnt/ngv/askc-home/SensorTransfer_datasets/GTA_repro10k/VOC2012/JPEGImages_ccrop512x1024:/root/dataset_synth/VOC2012/JPEGImages:ro \
  -v `pwd`/pretrained-networks/tensorflow-vgg:/mnt/ngv/pretrained-networks/tensorflow-vgg:ro \
  -v `pwd`/src:/root/cityscapes-fcn8s:ro \
  -v `pwd`/out-gtaSim10k2city-STgen-4convlayer-fcnstyleloss-voc-4epoch:/root/out \
  -v `pwd`/mean-images:/root/data \
  sensor-transfer-tf-docker \
  python3 main_STgen_fcnstyleloss_4convlayers_voc_orig.py \
    --data_dir_synth /root/dataset_synth \
    --data_dir_real /root/dataset_real \
    --data_slug_synth gta_sim10k_voc_crop \
    --data_slug_real cityscapes_voc_jpg \
    --phase augment \
    --num_epochs 4\
    --learning_rate 2e-5 \
    --gpu 2,3 
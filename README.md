# SensorEffectTransferNetwork
This is the public git repo for the paper "Sensor Transfer: Learning Optimal Sensor Effect Image Augmentation for Sim-to-Real Domain Adaptation"

## Requirements and Installation
This code base uses Docker. To install docker, please use the following: [https://docs.docker.com/install/]
To build the Docker Image, cd into the sensor transfer folder and run the following in the terminal:

`docker build -t sensor-transfer-tf-docker . `

There are also a good number of high quality docker tutorials provided on the docker website if you would like further reading.

You will also need to download the network weights into pretrained/tensorflow-vgg/ folder. You can download these weights (`vgg16.npy`) from [https://github.com/machrisaa/tensorflow-vgg]


## Running the SensorTransfer Network
The two bash files, `train-gta2city-STgen-styleloss-aug-voc.sh` and `train-gta2kitti-STgen-styleloss-aug-voc.sh`, will run the sensor transfer network model for translating GTA to Cityscapes and GTA to KITTI, respectively.
The anatomy of the bash file(s) is shown below:

```
nvidia-docker run -it \
  -v /path/to/JPEGImages:/root/dataset_real/VOC2012/JPEGImages:ro \
  -v /path/to/synthetic/JPEGImages:/root/dataset_synth/VOC2012/JPEGImages:ro \
  -v `pwd`/pretrained-networks/tensorflow-vgg:/mnt/ngv/pretrained-networks/tensorflow-vgg:ro \
  -v `pwd`/src:/root/cityscapes-fcn8s:ro \
  -v `pwd`/out-synthetic2real-weights-and-augimages:/root/out \
  -v `pwd`/mean-images:/root/data \
  sensor-transfer-tf-docker \
  python3 main_STgen_fcnstyleloss_voc_orig.py \
    --data_dir_synth /root/dataset_synth \
    --data_dir_real /root/dataset_real \
    --data_slug_synth gta_sim10k_voc_crop \
    --data_slug_real cityscapes_voc_jpg \
    --phase augment \
    --num_epochs 4\
    --learning_rate 2e-5 \
    --gpu 2,3 
```

The first two docker volumes map in the real and synthetic datasets. 

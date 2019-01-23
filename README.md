# SensorEffectTransferNetwork
This is the public git repo for the paper "Sensor Transfer: Learning Optimal Sensor Effect Image Augmentation for Sim-to-Real Domain Adaptation"

## Installation
This code base uses Docker. To install docker, please use the following: [https://docs.docker.com/install/]
To build the Docker Image, cd into the sensor transfer folder and run the following in the terminal:
`docker build -t sensor-transfer-tf-docker .`


You will also need to download the network weights into pretrained/tensorflow-vgg/ folder. You can download these weights (`vgg16.npy`) from [https://github.com/machrisaa/tensorflow-vgg]

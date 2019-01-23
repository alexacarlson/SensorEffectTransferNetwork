FROM tensorflow/tensorflow:latest-devel-gpu-py3

RUN apt-get update && \
    apt-get -y install python3-pillow

RUN cd /root && git clone https://github.com/umautobots/cityscapesScripts.git -b multi-processing-eval && \
    cd cityscapesScripts && \
    python3 -m pip install cython && \
    python3 setup.py build_ext --inplace

COPY src /root/cityscapes-fcn8s

WORKDIR /root/cityscapes-fcn8s
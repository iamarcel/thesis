FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y \
    libgtk-3-dev \
    dbus-x11 \
    build-essential \
    git \
    cmake \
    libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler \
    libboost-all-dev \
    libgflags-dev libgoogle-glog-dev liblmdb-dev \
    libopencv-dev \
    python-pip

ENV CAFFE_VERSION 0.15
LABEL com.nvidia.caffe.version="0.15"
ENV CAFFE_PKG_VERSION 0.15.13-1ubuntu16.04+cuda8.0

RUN apt-get install -y --no-install-recommends \
  caffe-nv=$CAFFE_PKG_VERSION \
  caffe-nv-tools=$CAFFE_PKG_VERSION \
  python-caffe-nv=$CAFFE_PKG_VERSION \
  libcaffe-nv0=$CAFFE_PKG_VERSION

RUN pip install --upgrade \
    numpy protobuf

WORKDIR /root
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose && \
    cd openpose && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . -- -j$(nproc)

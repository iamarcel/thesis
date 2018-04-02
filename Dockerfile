FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y \
  cuda-cublas-9-0 \
  torch7-nv \
	libhdf5-dev \
  libgtk-3-dev \
  dbus-x11 \
  ffmpeg

RUN apt-get install -y \
	python3 \
  python3-cairo \
	python3-dev \
	python3-pip \
  python3-gi \
  python3-gi-cairo \
  python3-tk \
  python-pil \
  gir1.2-gtk-3.0 \
  gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 \
  libgstreamer1.0-0 \
  libgstreamer1.0-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip3 install --upgrade pip
RUN pip3 install \
	h5py \
	jupyter \
	matplotlib \
	numpy \
	pandas \
	scipy \
	sklearn \
	six \
  Pillow \
	tensorflow-gpu \
  opencv-python \
  imageio

ADD ./src /root/dev
WORKDIR /root/dev

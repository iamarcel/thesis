FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update && apt-get install -y \
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
	gstreamer1.0-libav \
	ffmpeg \
	unzip \
  tmux
  # libav-tools \
  # libfreeimage3 \
  # libgl1-mesa-dev \
  # libjpeg8-dev \
  # libpci3 \
  # libxaw7 \
  # libzzip-0-13 \
  # libssh-dev \
  # libzip-dev \
  # mesa-common-dev \
  # xserver-xorg-core \
  # libx11-dev \
  # libxslt1.1 \
  # libpulse-mainloop-glib0

# Install correct Protobuf version for TensorFlow
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip && \
  unzip protoc-3.5.1-linux-x86_64.zip -d protoc3 && \
  mv protoc3/bin/* /usr/local/bin/ && \
  mv protoc3/include/* /usr/local/include/

# Install NAO dependencies
# ADD pynaoqi-python2.7-2.5.5.5-linux64.tar.gz /root/
# ADD webots_2018a-rev2_amd64.deb /root/webots_2018a-rev2_amd64.deb
# RUN dpkg -i /root/webots_2018a-rev2_amd64.deb && \
#   rm /root/webots_2018a-rev2_amd64.deb && \
#   echo 'export PYTHONPATH=${PYTHONPATH}:/root/pynaoqi-python2.7-2.5.5.5-linux64'

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
  pip3 install --upgrade 'pip<10.0.0' && pip3 install \
	  h5py \
	  jsonlines \
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
	  imageio \
    html5lib==0.999999999 && \
  mkdir -p /root/.jupyter/nbconfig && \
	echo '{ "CodeCell": { "cm_config": { "indentUnit": 2 } } }' > /root/.jupyter/nbconfig/notebook.json

ADD ./src /root/dev
WORKDIR /root/dev
EXPOSE 8888
EXPOSE 6006

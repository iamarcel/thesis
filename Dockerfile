FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update && apt-get install -y \
	python \
	python-cairo \
	python-dev \
	python-pip \
	python-gi \
	python-gi-cairo \
	python-tk \
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

ADD ./Pipfile.lock /root/dev/Pipfile.lock
WORKDIR /root/dev

RUN echo "export TFHUB_CACHE_DIR=/root/dev/tfhub" >> /root/.bashrc && \
  pip install --upgrade 'pip<10.0.0' && \
  pip install pipenv && \
  mkdir -p /root/.jupyter/nbconfig && \
	echo '{ "CodeCell": { "cm_config": { "indentUnit": 2 } } }' > /root/.jupyter/nbconfig/notebook.json && \
  cd /root/dev && \
  pipenv install --system --deploy --ignore-pipfile

EXPOSE 8888
EXPOSE 6006

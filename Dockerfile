FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y torch7-nv
RUN apt-get install -y xorg xauth

ENV display :0

ADD ./src /root/dev

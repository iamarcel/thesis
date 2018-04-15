PWD=$(shell pwd)
XSOCK=/tmp/.X11-unix

start: Dockerfile
	xhost +local:root
	sudo docker run --runtime=nvidia -it \
		-e "DISPLAY" \
		-v $(PWD)/src:/root/dev \
		-v $(XSOCK):$(XSOCK) \
		--device /dev/snd \
		-e PULSE_SERVER=unix:$(XDG_RUNTIME_DIR)/pulse/native \
		-v $(XDG_RUNTIME_DIR)/pulse/native:$(XDG_RUNTIME_DIR)/pulse/native \
		--group-add $(shell getent group audio | cut -d: -f3) \
		-p 8888:8888 \
		-p 6006:6006 \
		thesis
	xhost -local:root

build:
	sudo docker build -t thesis .

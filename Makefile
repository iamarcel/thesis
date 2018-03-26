PWD=$(shell pwd)
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

start:
	touch $(XAUTH)
	xauth nlist $(DISPLAY) | sed -e 's/^..../ffff/' | xauth -f $(XAUTH) nmerge -
	chmod 755 $(XAUTH)
	sudo docker run -it \
		-v $(PWD)/src:/root/dev \
		-v $(XSOCK):$(XSOCK) \
		-v $(XAUTH):$(XAUTH) \
		-e XAUTHORITY=$(XAUTH) \
		thesis

build:
	sudo docker build -t thesis .

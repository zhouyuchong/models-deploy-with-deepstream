CUDA_VER?=11.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -Wno-error=deprecated-declarations
CFLAGS+= -shared -fPIC

NVDS_VERSION:=6.1
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsparse_yolov5.cpp
TARGET_LIB:= libnvdsinfer_custom_impl_yolo_fire.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)

clean:
	rm -rf $(TARGET_LIB)

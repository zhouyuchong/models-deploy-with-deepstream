CUDA_VER?=11.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -g -std=c++11 -shared -fPIC
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes/ -I/usr/local/cuda-$(CUDA_VER)/include
#CFLAGS+=-I../../includes -I/home/dell/Pictures/TensorRT-7.0.0.11/include
#CFLAGS+=-I../../includes -I/opt/nvidia/deepstream/deepstream-5.0/sources/includes
LIBS:= -L/usr/local/cuda-11.6/lib64 -L/usr/lib/x86_64-linux-gnu -L/opt/nvidia/deepstream/deepstream/lib/ -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas 
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

# CFLAGS:= -Wall -Werror -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations

# CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes

# LIBS:= -lnvinfer -lnvparsers
# LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvinfer_custom_genderage_parser.cpp
TARGET_LIB:= libnvdsinfer_custom_impl_genderage.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
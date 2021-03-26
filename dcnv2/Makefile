#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-03-25 19:05:52
#***
#************************************************************************************/

# Location of the CUDA Toolkit
CUDA_ROOT_PATH ?= /usr/local/cuda-10.2
# TORCH_ROOT_PATH ?=/home/dell/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch
TORCH_ROOT_PATH ?=/opt/libtorch

LIBNAME := libdcnv2

CXXFLAGS := -O3 -Wall -fPIC
DEFINES := \
	-DNDEBUG -DWITH_CUDA \
	-DTORCH_API_INCLUDE_EXTENSION_H \
	-DTORCH_EXTENSION \
	-D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14


INCLUDES := \
	-I${TORCH_ROOT_PATH}/include \
	-I${TORCH_ROOT_PATH}/include/torch/csrc/api/include \
	-I${TORCH_ROOT_PATH}/include/TH \
	-I${TORCH_ROOT_PATH}/include/THC \
	-I${CUDA_ROOT_PATH}/include \
	-I. \

#   --sysroot=<directory>    Use <directory> as the root directory for headers and libraries.
#   -Wl,<options>            Pass comma-separated <options> on to the linker.
#   -B <directory>           Add <directory> to the compiler's search paths.

LDFLAGS     := \
	-B ${TORCH_ROOT_PATH}/compiler_compat \
	-L${TORCH_ROOT_PATH}/lib -Wl,-rpath=${TORCH_ROOT_PATH}/lib \
	-Wl,--no-as-needed -Wl,--sysroot=/

LIBRARIES := \
	-L${TORCH_ROOT_PATH}/lib \
	-L${CUDA_ROOT_PATH}/lib64 \
	-lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda -lcudart

SOURCE :=  \
	deform_conv2d.cpp \
	cpu/deform_conv2d_kernel.cpp \
	cuda/deform_conv2d_kernel.cu


################################################################################


##############################
TARGET_SIZE := 64

NVCC          := $(CUDA_ROOT_PATH)/bin/nvcc

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} \
	-DCUDA_HAS_FP16=1 \
	-D__CUDA_NO_HALF_OPERATORS__ \
	-D__CUDA_NO_HALF_CONVERSIONS__ \
	-D__CUDA_NO_HALF2_OPERATORS__ \
	--expt-relaxed-constexpr \
	--compiler-options '-fPIC' \
	-arch sm_75 \

NVCCFLAGS += $(EXTRA_NVCCFLAGS)
NVCCFLAGS += $(addprefix -Xcompiler ,$(CXXFLAGS))
NVCCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CXXFLAGS))

CXXFLAGS := ${CXXFLAGS}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))


#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) ${DEFINES} $(INCLUDES) -o $@ -c $<

%.o : %.cpp
	${CXX} ${CXXFLAGS} ${DEFINES} ${INCLUDES} -c $< -o $@

%.o : %.c
	${CC} ${CFLAGS} ${INCS} -c $< -o $@


#****************************************************************************
# Target rules
#****************************************************************************
all: staticlib

sharelib: $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -o $(LIBNAME).so $(OBJECTS) $(LIBRARIES)

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBNAME).a $(OBJECTS)

clean:
	rm -f ${OBJECTS} ${LIBNAME}.so ${LIBNAME}.a

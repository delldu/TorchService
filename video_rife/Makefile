#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-11-22 02:26:07
#***
#************************************************************************************/
#
INSTALL_DIR := /usr/local
TORCH_INSTALL_DIR := /opt/libtorch

TARGET := video_rife

SOURCE :=  \
	main.c \

INCS	:= -I../lib -isystem$(TORCH_INSTALL_DIR)/include $(shell nimagetool --cflags)

LIBS	:= -L../lib -ltorcheng \
	-Wl,--no-as-needed -L${TORCH_INSTALL_DIR}/lib -lc10 -lc10_cuda -ltorch -ltorch_cuda -ltorch_cpu \
	$(shell nimagetool --libs)

DEFINES := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14

CXXFLAGS := -O3 -fPIC -Wall -Wextra
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	sudo install -m 755 ${TARGET} ${INSTALL_DIR}/bin
	sudo cp video_rife.service /lib/systemd/system/
	sudo systemctl daemon-reload


#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}

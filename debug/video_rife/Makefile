#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-11-22 02:26:07
#***
#************************************************************************************/
#
INSTALL_DIR := /usr/local

TARGET := video_rife

SOURCE :=  \
	main.c \

INCS	:= -I../lib $(shell torchtool --cflags) $(shell nimagetool --cflags)

LIBS	:= -L../lib -ltorcheng \
	$(shell torchtool --libs) \
	$(shell nimagetool --libs)

DEFINES :=

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
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin
	cp video_rife.service /lib/systemd/system/
	systemctl daemon-reload


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

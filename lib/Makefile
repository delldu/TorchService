#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-02-07 16:51:15
#***
#************************************************************************************/
#

LIB_NAME := libtorcheng

INCS	:= $(shell torchtool --cflags) $(shell nimagetool --cflags)

SOURCE :=  \
	engine.c

DEFINES :=

CXXFLAGS := -fPIC -Wall -Wextra
LDFLAGS := -fPIC -O3

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB_NAME).a $(OBJECTS)


#****************************************************************************
# Depend block
#****************************************************************************
depend:

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


clean:
	# DO NOT DELETE libdcnv2.so
	rm -rf *.a libtorcheng.so *.o $(OBJECTS)


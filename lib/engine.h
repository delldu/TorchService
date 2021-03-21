/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-03-20 21:19:01
***
************************************************************************************/

#ifndef _ENGINE_H
#define _ENGINE_H

#include <stdio.h>
#include <stdlib.h>

#include <nimage/image.h>
#include <nimage/nnmsg.h>

// One-stop header.
#include <torch/script.h>
typedef torch::jit::script::Module TorchModule;

// Torch Runtime Engine
typedef struct {
	DWORD magic;
	const char *model_path;
	TorchModule module;
	int use_gpu;
} TorchEngine;

#define CheckEngine(e) \
    do { \
            if (! ValidEngine(e)) { \
				fprintf(stderr, "Bad TorchEngine.\n"); \
				exit(1); \
            } \
    } while(0)

TorchEngine *CreateEngine(const char *model_path, int use_gpu);
int ValidEngine(TorchEngine * engine);
TENSOR *TensorForward(TorchEngine * engine, TENSOR * input);
void DestroyEngine(TorchEngine * engine);

int TorchService(char *endpoint, char *onnx_file, int use_gpu);
TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, int *rescode);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);


#endif // _ENGINE_H

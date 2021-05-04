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

#include <vision_service.h>

// One-stop header.
#include <torch/script.h>

typedef at::Tensor AtTensor;
typedef torch::Tensor TorchTensor;
typedef torch::jit::script::Module TorchModule;

static TIME engine_last_running_time = 0;


// Torch Runtime Engine
typedef struct {
	DWORD magic;
	char *model_path;
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

// typedef int (*CustomSevice)(int socket, int service_code, TENSOR *input_tensor);
typedef int (*CustomSevice)(int, int, TENSOR *);


TorchEngine *CreateEngine(char *model_path, int use_gpu);
int ValidEngine(TorchEngine * engine);
TENSOR *TensorForward(TorchEngine * engine, TENSOR * input);
void DestroyEngine(TorchEngine * engine);

int TorchService(char *endpoint, char *torch_file, int service_code, int use_gpu, CustomSevice custom_service_function);
TENSOR *OnnxRPC(int socket, TENSOR * input, int seqcode, int *rescode);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);


#define ENGINE_IDLE_TIME (120*1000)	// 120 seconds

#define StartEngine(engine, torch_file, use_gpu) \
do { \
	if (engine == NULL) \
		engine = CreateEngine(torch_file, use_gpu); \
	CheckEngine(engine); \
	engine_last_running_time = time_now(); \
} while(0)

#define StopEngine(engine) \
do { \
	engine_last_running_time = time_now(); \
	DestroyEngine(engine); \
	engine = NULL; \
} while(0)

#define EngineIsIdle() (time_now() - engine_last_running_time > ENGINE_IDLE_TIME)


#endif // _ENGINE_H

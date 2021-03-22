/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-03-20 23:52:44
***
************************************************************************************/

// Reference: https://pytorch.org/tutorials/advanced/cpp_export.html

#include <assert.h>
#include <vector>
#include <memory>


// For mkdir
#include <sys/types.h>
#include <sys/stat.h> 
#include "engine.h"

typedef torch::Tensor TorchTensor;

// Torch Runtime Engine
#define MAKE_FOURCC(a,b,c,d) (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) | ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('T', 'O', 'R', 'C')

TorchEngine *CreateEngine(const char *model_path, int use_gpu)
{
	TorchEngine *t;

	syslog_info("Creating Torch Runtime Engine for model %s ...", model_path);

	t = (TorchEngine *) calloc((size_t) 1, sizeof(TorchEngine));
	if (!t) {
		syslog_error("Allocate memeory.");
		return NULL;
	}
	t->magic = ENGINE_MAGIC;
	t->model_path = model_path;
	t->use_gpu = use_gpu;

	try {
		t->module = torch::jit::load(model_path);
	} catch(const c10::Error & e) {
		syslog_error("Loading model %s", model_path);
		free(t);
		return NULL;
	}

	CheckPoint("Module size = %d", sizeof(t->module));

	// to GPU
	if (use_gpu)
		t->module.to(at::kCUDA);

	syslog_info("Create Torch Runtime Engine OK.");

	return t;
}

int ValidEngine(TorchEngine * t)
{
	return (!t || t->magic != ENGINE_MAGIC) ? 0 : 1;
}

TENSOR *TensorForward(TorchEngine * engine, TENSOR * input)
{
	int b, c, i, j, n;
	int dims[4];
	float *data;
	TENSOR *output = NULL;

	CHECK_TENSOR(input);
	auto input_tensor = torch::from_blob(input->data, 
			{input->batch, input->chan, input->height, input->width});

	// std::cout << "input dimensions: " << input_tensor.sizes() << std::endl;
	if (engine->use_gpu)
		input_tensor = input_tensor.to(torch::kCUDA);
	TorchTensor output_tensor = engine->module.forward({input_tensor}).toTensor();
	if (engine->use_gpu)
		output_tensor = output_tensor.to(torch::kCPU);
	// std::cout << "Output dimensions: " << output_tensor.sizes() << std::endl;

	// torch::Tensor tensor = torch::randn({3, 4, 5});
	// assert(tensor.sizes() == std::vector<int64_t>{3, 4, 5});
	auto output_tensor_dims = output_tensor.sizes();
	n = 4 - output_tensor_dims.size();
	for (i = 0; i < n; i++)
		dims[i] = 1;
	for (i = n; i < 4; i++)
		dims[i] = output_tensor_dims[i - n];

	output = tensor_create(dims[0], dims[1], dims[2], dims[3]);
	CHECK_TENSOR(output);

	// auto f_a = output_tensor.accessor<float, 4>();
	float *f = (float *)output_tensor.data_ptr();
	for (b = 0; b < dims[0]; b++) {
		for (c = 0; c < dims[1]; c++) {
			data = tensor_start_chan(output, b, c);
			for (i = 0; i < dims[2]; i++) {
				for (j = 0; j < dims[3]; j++) {
					*data++ = *f++;
				}
			}
		}
	}

	// delete input_tensor/output_tensor ?;
	// if (engine->use_gpu) {
	// }

	return output;
}

void DestroyEngine(TorchEngine * engine)
{
	if (!ValidEngine(engine))
		return;

	// delete engine->module ?;
	// if (engine->use_gpu) {
	// }

	free(engine);
}

int TorchService(char *endpoint, char *onnx_file, int use_gpu)
{
	int socket, reqcode, count, rescode;
	TENSOR *input_tensor, *output_tensor;
	TorchEngine *engine;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	engine = CreateEngine(onnx_file, use_gpu);
	CheckEngine(engine);

	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

		input_tensor = request_recv(socket, &reqcode);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = %d", reqcode);

		// Real service ...
		time_reset();
		output_tensor = TensorForward(engine, input_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(input_tensor);

		count++;
	}
	DestroyEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, int *rescode)
{
	TENSOR *output = NULL;

	CHECK_TENSOR(input);

	if (request_send(socket, reqcode, input) == RET_OK) {
		output = response_recv(socket, rescode);
	}

	return output;
}

void SaveOutputImage(IMAGE *image, char *filename)
{
	char output_filename[256], *p;

	mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 

	if (image_valid(image)) {
		p = strrchr(filename, '/');
	 	p = (! p)? filename : p + 1;
		snprintf(output_filename, sizeof(output_filename) - 1, "output/%s", p);
		image_save(image, output_filename);
	}
}

void SaveTensorAsImage(TENSOR *tensor, char *filename)
{
	IMAGE *image = image_from_tensor(tensor, 0);

	if (image_valid(image)) {
		SaveOutputImage(image, filename);
		image_destroy(image);
	}
}

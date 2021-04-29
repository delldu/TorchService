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

#include <cuda_runtime_api.h>	// locate /usr/local/cuda-10.2/include for cudaDeviceReset
// #include <c10/cuda/CUDACachingAllocator.h>	// for emptyCache
static TIME engine_last_running_time = 0;

typedef torch::Tensor TorchTensor;

// Torch Runtime Engine
#define MAKE_FOURCC(a,b,c,d) (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) | ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('T', 'O', 'R', 'C')

char *FindModel(char *modelname);


TorchEngine *CreateEngine(char *model_path, int use_gpu)
{
	TorchEngine *t;

	syslog_info("Creating Torch Runtime Engine for model %s ...", model_path);

	CheckPoint("Model load before");
	system("nvidia-smi");

	t = (TorchEngine *) calloc((size_t) 1, sizeof(TorchEngine));
	if (!t) {
		syslog_error("Allocate memeory.");
		return NULL;
	}
	t->magic = ENGINE_MAGIC;
	t->model_path = FindModel(model_path);
	t->use_gpu = use_gpu;

	try {
		t->module = torch::jit::load(t->model_path);
	} catch(const c10::Error & e) {
		syslog_error("Loading model %s", t->model_path);
		free(t);
		return NULL;
	}

	CheckPoint("Model load after");
	system("nvidia-smi");

	// CheckPoint("Module size = %d", sizeof(t->module)); ==> 8

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
	int b, c, i, n;
	int dims[4];
	float *data;
	TENSOR *output = NULL;

	CheckPoint("Forward before");
	system("nvidia-smi");

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
			for (i = 0; i < dims[2] * dims[3]; i++)
				*data++ = *f++;
		}
	}

	CheckPoint("Forward after");
	system("nvidia-smi");

	// delete input_tensor/output_tensor ?;
	if (engine->use_gpu) {
		// DO NOT DO !!! following, will cause crash !!!
		// cudaFree(input_tensor.storage().data());
		// cudaFree(input_tensor.storage().data());

		// The following has no effect
		//c10::cuda::CUDACachingAllocator::emptyCache();
	}

	return output;
}

void DestroyEngine(TorchEngine * engine)
{
	if (!ValidEngine(engine))
		return;

	if (engine->model_path)
		free(engine->model_path);

	// delete engine->module ?;
	if (engine->use_gpu)
		cudaDeviceReset();

	free(engine);
}

int TorchService(char *endpoint, char *torch_file, int service_code, int use_gpu, CustomSevice custom_service_function)
{
	int socket, msgcode, count;
	TENSOR *input_tensor, *output_tensor;
	TorchEngine *engine = NULL;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	if (! custom_service_function)
		custom_service_function = service_response;

	count = 0;
	for (;;) {
		if (EngineIsIdle())
			StopEngine(engine);

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (! tensor_valid(input_tensor))
			continue;

		if (msgcode == service_code) {
			syslog_info("Service %d times", count);
			StartEngine(engine, onnx_file, use_gpu);

			// Real service ...
			time_reset();
			output_tensor = TensorForward(engine, input_tensor);
			time_spend((char *)"Predict");

			service_response(socket, service_code, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else {
			// service_response(socket, servicecode, input_tensor)
			custom_service_function(socket, OUTOF_SERVICE, NULL);
		}

		tensor_destroy(input_tensor);
	}
	StopEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, int *rescode)
{
	TENSOR *output = NULL;

	CHECK_TENSOR(input);

	*rescode = 0;
	if (tensor_send(socket, reqcode, input) == RET_OK) {
		output = tensor_recv(socket, rescode);
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


char *FindModel(char *modelname)
{
	char filename[256];

	snprintf(filename, sizeof(filename), "%s", modelname);
	if(access(filename, F_OK) == 0) {
		CheckPoint("Found Model: %s", filename);
		return strdup(filename);
	}

	snprintf(filename, sizeof(filename), "%s/%s", TORCHMODEL_INSTALL_DIR, modelname);
	if(access(filename, F_OK) == 0) {
		CheckPoint("Found Model: %s", filename);
		return strdup(filename);
	}

	syslog_error("Model %s NOT Found !", modelname);
	return NULL;
}

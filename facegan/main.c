/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-11-22 13:18:11
***
************************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <syslog.h>

#include <nimage/image.h>
#include <nimage/nnmsg.h>

#include "engine.h"

#include <random>	// Support since C++11 
#include <torch/torch.h>
// #include <iostream>


#define W_SPACE_DIM 512


TorchEngine *trans_engine = NULL;	
TorchEngine *decoder_engine = NULL;
TorchEngine *loss_engine = NULL;

TENSOR *reference_face = NULL;

// Fill normal distribution to zcode
int normal_zdata(TENSOR *zcode_tensor)
{
	int i;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dis{0.0, 1.0};	// mean = 0.0, std = 1.0

	srand(time(NULL));
	check_tensor(zcode_tensor);

	for (i = 0; i < W_SPACE_DIM; i++)
		zcode_tensor->data[i] = dis(gen);

	return RET_OK;
}

TENSOR *random_zcode()
{
	TENSOR *zcode_tensor;

	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);
	
	normal_zdata(zcode_tensor);

	return zcode_tensor;
}

TENSOR *random_wcode()
{
	TENSOR *zcode_tensor, *wcode_tensor;

	CheckEngine(trans_engine);

	// Create zcode tensor
	zcode_tensor = random_zcode();
	CHECK_TENSOR(zcode_tensor);

	// Compute wcode
	wcode_tensor = TensorForward(trans_engine, zcode_tensor);
	CHECK_TENSOR(wcode_tensor);

	tensor_destroy(zcode_tensor);

	return wcode_tensor;
}

/*****************************************************
*
* Start optimizing from mean wcode
*
******************************************************/
TENSOR *mean_wcode()
{
	int i, j, n;
	TENSOR *zcode_tensor, *wcode_tensor, *average_tensor;

	CheckEngine(trans_engine);

	average_tensor = tensor_create(1, 18, 1, W_SPACE_DIM);
	CHECK_TENSOR(average_tensor);
	memset(average_tensor->data, 0, 18 * W_SPACE_DIM * sizeof(float));

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);

	n = 4096;
	for (i = 0; i < n; i++) {
		normal_zdata(zcode_tensor);

		wcode_tensor = TensorForward(trans_engine, zcode_tensor);
		CHECK_TENSOR(wcode_tensor);

		// Save wcode_tensor ...
		for (j = 0; j < 18 * W_SPACE_DIM; j++)
			average_tensor->data[i] += wcode_tensor->data[i];

		tensor_destroy(wcode_tensor);
	}

	tensor_destroy(zcode_tensor);

	for (j = 0; j < 18 * W_SPACE_DIM; j++)
		average_tensor->data[i] /= n;

	return average_tensor;
}

int save_reference(TENSOR *input_tensor)
{
	int i, n;
	check_tensor(input_tensor);

	// Normal for perception loss
	n = input_tensor->batch * input_tensor->chan * input_tensor->height * input_tensor->width;
	for (i = 0; i < n; i++)
		input_tensor->data[i] -= 0.5;

	reference_face = input_tensor;

	return RET_OK;
}

void release_reference()
{
	reference_face = NULL;	// Just set to null, nothing to do.
}

void gradient_descent(AtTensor& x, AtTensor& grad, float lr)
{
	int i, n;
	float *x_data, *grad_data;

	auto dims = x.sizes();
	for (n = 1, i = 0; i < (int)dims.size(); i++)
		n *= dims[i];

	x_data = (float *)x.data_ptr();
	grad_data = (float *)grad.data_ptr();
	for (i = 0; i < n; i++)
		x_data[i] -= lr * grad_data[i];
}


TENSOR *do_search(TENSOR *input_tensor)
{
	TENSOR *wcode_tensor, *image_tensor;	
	CHECK_TENSOR(input_tensor);

	wcode_tensor = random_wcode();
	CHECK_TENSOR(wcode_tensor);

	image_tensor = TensorForward(decoder_engine, wcode_tensor);
	tensor_destroy(wcode_tensor);

	return image_tensor;
}

TENSOR *do_optimizing(int epochs, float lr)
{
	int i, index;
	float *f;
	AtTensor loss, grad;
	TorchTensor image_tensor, reference_tensor;

	// One of the applications of higher-order gradients is calculating gradient penalty.
	// Let's see an example of it using ``torch::autograd::grad``:

	TENSOR *mean = mean_wcode();
	CHECK_TENSOR(mean);
	CheckEngine(decoder_engine);

	// Create reference_tensor for compare
	reference_tensor = torch::zeros({1, 3, 256, 256});
	f = (float *)reference_tensor.data_ptr();
	for (i = 0; i < 1 * 3 * 256 * 256; i++)
		f[i] = reference_face->data[i];



	AtTensor input_tensor = torch::from_blob(mean->data, 
			{mean->batch, mean->chan, mean->height, mean->width}).requires_grad_(true);

	index = 0;
	while(index < epochs) {
		image_tensor = decoder_engine->module.forward({input_tensor}).toTensor();

		// reshape(batch, channel, height // factor, factor, width // factor, factor)
		image_tensor = image_tensor.reshape({1, 3, 256, 4, 256, 4});
		image_tensor = image_tensor.mean({3, 5});

		loss = torch::nn::MSELoss()(image_tensor, reference_tensor);

		if (loss.item<float>() < 1e-3f)
			break;

		loss.backward();

		grad = input_tensor.grad();
		gradient_descent(input_tensor, grad, lr);

		index++;
	}




	auto model = torch::nn::Linear(4, 3);

	auto input = torch::randn({3, 4}).requires_grad_(true);
	auto output = model(input);

	// Calculate loss
	auto target = torch::randn({3, 3});

	// Use norm of gradients as penalty
	auto grad_output = torch::ones_like(output);
	auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
	auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

	// Add gradient penalty to loss
	// auto combined_loss = loss + gradient_penalty;
	// combined_loss.backward();

	std::cout << input.grad() << std::endl;

	// -0.0260  0.1660  0.3094  0.0113
	//  0.1350  0.2266  0.1991 -0.0556
	//  0.1171  0.1380 -0.0355 -0.0320

	return NULL;
}




int FaceGanService(char *endpoint, int use_gpu, CustomSevice custom_service_function)
{
	int socket, msgcode, count;
	TENSOR *input_tensor, *image_tensor;


	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	if (! custom_service_function)
		custom_service_function = service_response;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(trans_engine);
			StopEngine(decoder_engine);
			// StopEngine(loss_engine);
		}

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (! tensor_valid(input_tensor))
			continue;

		if (msgcode == IMAGE_FACEGAN_SERVICE) {
			syslog_info("Service %d times", count);

			StartEngine(trans_engine, (char *)"FaceganTransformer.pt", use_gpu);
			StartEngine(decoder_engine, (char *)"FaceganDecoder.pt", use_gpu);
			// StartEngine(loss_engine, (char *)"FaceganLoss.pt", use_gpu);

			// Real service ...
			time_reset();
			save_reference(input_tensor);
			image_tensor = do_search(input_tensor);
			time_spend((char *)"Searching");

			service_response(socket, IMAGE_FACEGAN_SERVICE, image_tensor);
			tensor_destroy(image_tensor);

			count++;
		} else {
			// service_response(socket, servicecode, input_tensor)
			custom_service_function(socket, OUTOF_SERVICE, NULL);
		}

		tensor_destroy(input_tensor);

		release_reference();
	}
	StopEngine(trans_engine);
	StopEngine(decoder_engine);
	// StopEngine(loss_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}


int server(char *endpoint, int use_gpu)
{
	return FaceGanService(endpoint, use_gpu, NULL);
}

TENSOR *load_tensor(char *input_file)
{
	IMAGE *image;
	TENSOR *tensor;

	image = image_load(input_file); CHECK_IMAGE(image);

	tensor = tensor_from_image(image, 0); CHECK_TENSOR(tensor);

	image_destroy(image);

	return tensor;
}

TENSOR *facegan_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw, rescode;
	TENSOR *resize_send, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	nh = nw = 256;
	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_FACEGAN_SERVICE, &rescode);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);
		recv_tensor = OnnxRPC(socket, resize_send, IMAGE_FACEGAN_SERVICE, &rescode);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}


int facegan(int socket, char *input_file)
{
	TENSOR *send_tensor, *recv_tensor;

	printf("Face generating %s ...\n", input_file);

	send_tensor = load_tensor(input_file);
	if (tensor_valid(send_tensor)) {
		recv_tensor = facegan_onnxrpc(socket, send_tensor);
		if (tensor_valid(recv_tensor)) {
			SaveTensorAsImage(recv_tensor, input_file);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
	}

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option] <image files -- input ... >\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server <0 | 1>         Start server (use gpu).\n");

	exit(1);
}


int main(int argc, char **argv)
{
	int i, optc;
	int use_gpu = 1;
	int running_server = 0;
	int socket;

	int option_index = 0;
	char *endpoint = (char *) IMAGE_FACEGAN_URL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 1, 0, 's'},
		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
			break;
		case 'h':				// help
		default:
			help(argv[0]);
			break;
		}
	}

	if (running_server)
		return server(endpoint, use_gpu);
	else if (argc > optind) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i < argc; i++)
			facegan(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}

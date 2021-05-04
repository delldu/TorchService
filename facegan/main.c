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
#include <iostream>

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

	average_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(average_tensor);
	memset(average_tensor->data, 0, 1 * W_SPACE_DIM * sizeof(float));

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);

	n = 1; // xxxx8888
	for (i = 0; i < n; i++) {
		normal_zdata(zcode_tensor);

		wcode_tensor = TensorForward(trans_engine, zcode_tensor);
		CHECK_TENSOR(wcode_tensor);
		// Save wcode_tensor ...
		for (j = 0; j < W_SPACE_DIM; j++)
			average_tensor->data[j] += wcode_tensor->data[j];

		tensor_destroy(wcode_tensor);
	}

	tensor_destroy(zcode_tensor);

	for (j = 0; j < W_SPACE_DIM; j++)
		average_tensor->data[i] /= n;

	return average_tensor;
}

int save_reference(TENSOR *input_tensor)
{
	// int i, n;
	check_tensor(input_tensor);

	// Normal for perception loss
	// n = input_tensor->batch * input_tensor->chan * input_tensor->height * input_tensor->width;
	// for (i = 0; i < n; i++)
	// 	input_tensor->data[i] -= 0.5;

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

// def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
//     lr_ramp = min(1, (1 - t) / rampdown)
//     lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
//     lr_ramp = lr_ramp * min(1, t / rampup)
//     return initial_lr * lr_ramp

float learning_rate(float t, float init_lr)
{
	float rampdown = 0.25;
	float rampup = 0.05;
	float lr_ramp = MIN(1.0, (1.0 - t)/rampdown);
	lr_ramp = 0.5 - 0.5 * cosf(lr_ramp * 3.1415926);
	lr_ramp = lr_ramp * MIN(1.0, t/rampup);

	return init_lr * lr_ramp;
}


TENSOR *best_wcode(int epochs, float lr)
{
	int i, index;
	TENSOR *best_tensor;
	float *fdata, best_loss, noise_strength, progress;
	TorchTensor latent_in, input_tensor, loss;
	TorchTensor image_tensor, reference_tensor;
	// AtTensor image_tensor, reference_tensor;

	CheckEngine(decoder_engine);
	CheckEngine(trans_engine);
	CheckEngine(loss_engine);

	// mean = mean_wcode(); CHECK_TENSOR(mean);

	// Create reference_tensor for compare
	reference_tensor = torch::zeros({1, 3, 256, 256});
	fdata = (float *)reference_tensor.data_ptr();
	for (i = 0; i < 1 * 3 * 256 * 256; i++)
		fdata[i] = reference_face->data[i];


	auto zcode = torch::randn({4096, W_SPACE_DIM});
	zcode = trans_engine->module.forward({zcode}).toTensor();
	latent_in = zcode.detach().mean(0, 1);
	latent_in.requires_grad_(true);

	CheckPoint("latent_in.is_leaf = %d", latent_in.is_leaf());

	torch::optim::Adam optimizer({latent_in}, lr=lr);
	// torch::optim::Adam optimizer({latent_in}, torch::optim::AdamOptions(lr).betas(std::make_tuple (0.5, 0.5)));

	index = 0;
	best_tensor = tensor_create(1, 1, 1, W_SPACE_DIM); CHECK_TENSOR(best_tensor);
	best_loss = 1000000000.0;

	while(index < epochs) {
		syslog_info("Searching epoch: %d/%d, best loss: %.4f", index + 1, epochs, best_loss);

		progress = 1.0 * index/epochs;
		noise_strength = pow(0.05 * MAX(0, 1 - progress/0.75), 2);

		input_tensor = latent_in + torch::randn_like(latent_in) * noise_strength;

		// Forward
		image_tensor = decoder_engine->module.forward({input_tensor}).toTensor();

		// reshape(batch, channel, height // factor, factor, width // factor, factor)
		image_tensor = image_tensor.reshape({1, 3, 256, 4, 256, 4});
		image_tensor = image_tensor.mean({3, 5});

		std::vector<torch::jit::IValue> loss_inputs;
    	loss_inputs.push_back(image_tensor);
    	loss_inputs.push_back(reference_tensor);

		loss = loss_engine->module.forward(loss_inputs).toTensor();
		// + 0.01 * torch::nn::MSELoss()(image_tensor, reference_tensor);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

		// Save best
		if (best_loss > loss.item<float>()) {
			best_loss = loss.item<float>();
			fdata = (float *)input_tensor.data_ptr();
			for (i = 0; i < W_SPACE_DIM; i++)
				best_tensor->data[i] = fdata[i];
		}

		if (best_loss < 1e-3f)
			break;

		// loss.backward();
		// grad = input_tensor.grad();
		// gradient_descent(input_tensor, grad, lr);
		// printf("Gradient: ");
		// fdata = (float *)grad.data_ptr();
		// for (i = 0; i < W_SPACE_DIM; i++)
		// 	printf("%.6f, ", fdata[i]);
		// printf("\n");

		index++;
	}

	return best_tensor;
}

TENSOR *do_search(TENSOR *input_tensor)
{
	TENSOR *wcode_tensor, *image_tensor;	
	CHECK_TENSOR(input_tensor);

	wcode_tensor = best_wcode(10, 0.1); // random_wcode();
	CHECK_TENSOR(wcode_tensor);

	image_tensor = TensorForward(decoder_engine, wcode_tensor);
	tensor_destroy(wcode_tensor);

	return image_tensor;
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
			StopEngine(loss_engine);
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
			StartEngine(loss_engine, (char *)"FaceganPercept.pt", use_gpu);

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
	StopEngine(loss_engine);

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

	// auto latent = torch::randn({4096, 512});
	// std::cout << "latent --- " << latent.sizes() << std::endl;
	// std::cout << "mean --- " << latent.mean(0, 1 /*keep dimmension */).sizes() << std::endl;
	// std::cout << "std --- " << latent.std(0).sizes() << std::endl;

	// latent.requires_grad_(true); 


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

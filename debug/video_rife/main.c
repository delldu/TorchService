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

int server(char *endpoint, int use_gpu)
{
	return TorchService(endpoint, (char *)"VideoRIFE.pt", VIDEO_RIFE_SERVICE, use_gpu, NULL);
}

TENSOR *blend_tensor(char *input_file1, char *input_file2)
{
	int n;
	IMAGE *image1, *image2;
	TENSOR *tensor1, *tensor2, *tensor;

	image1 = image_load(input_file1); CHECK_IMAGE(image1);
	image2 = image_load(input_file2); CHECK_IMAGE(image2);

	tensor1 = tensor_from_image(image1, 0); CHECK_TENSOR(tensor1);
	tensor2 = tensor_from_image(image2, 0); CHECK_TENSOR(tensor2);

	tensor = NULL;

	if (tensor1->height != tensor2->height || tensor1->width != tensor2->width) {
		syslog_error("Image is not same size between %s %s", input_file1, input_file2);
		goto failure;
	}

	n = 3 * tensor1->height * tensor1->width;
	tensor = tensor_create(2, 3, tensor1->height, tensor2->width);
	memcpy(tensor->data, tensor1->data, n * sizeof(float));
	memcpy(&tensor->data[n], tensor2->data, n * sizeof(float));

failure:

	tensor_destroy(tensor2);
	tensor_destroy(tensor1);
	image_destroy(image2);
	image_destroy(image1);

	return tensor;
}

TENSOR *rife_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw, rescode;
	TENSOR *resize_send, *resize_recv, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	// rife server limited: only accept 8 times tensor !!!
	// nh = (send_tensor->height + 7)/8; nh *= 8;
	// nw = (send_tensor->width + 7)/8; nw *= 8;
	space_resize(send_tensor->height, send_tensor->width, 1024, 8, &nh, &nw);

	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		recv_tensor = OnnxRPC(socket, send_tensor, VIDEO_RIFE_SERVICE, &rescode);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);
		resize_recv = OnnxRPC(socket, resize_send, VIDEO_RIFE_SERVICE, &rescode);
		recv_tensor = tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
		tensor_destroy(resize_recv);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}


int rife(int socket, char *input_file1, char *input_file2)
{
	TENSOR *send_tensor, *recv_tensor;

	printf("Interpolating %s %s ...\n", input_file1, input_file2);

	send_tensor = blend_tensor(input_file1, input_file2);
	if (tensor_valid(send_tensor)) {
		recv_tensor = rife_onnxrpc(socket, send_tensor);
		if (tensor_valid(recv_tensor)) {
			SaveTensorAsImage(recv_tensor, input_file1);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
	}

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option] <image files -- input1 input2 ... >\n", cmd);
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
	char *endpoint = (char *) VIDEO_RIFE_URL;

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
	else if (argc > optind + 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i + 1 < argc; i++)
			rife(socket, argv[i], argv[i + 1]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}

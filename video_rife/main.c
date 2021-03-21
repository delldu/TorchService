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

#define VIDEO_RIFE_REQCODE 0x0205
// #define VIDEO_RIFE_URL "ipc:///tmp/video_rife.ipc"
#define VIDEO_RIFE_URL "tcp://127.0.0.1:9205"

int server(char *endpoint, int use_gpu)
{
	return TorchService(endpoint, (char *)"video_rife.pt", use_gpu);
}

int rife(int socket, char *input_file)
{
	int rescode;
	IMAGE *orig_image, *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Interpolating %s ...\n", input_file);

	orig_image = image_load(input_file); check_image(orig_image);
	send_image = image_zoom(orig_image, 224, 224, 0); check_image(send_image);
	image_destroy(orig_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);
		check_tensor(send_tensor);

#if 1
		int i;
		float *data;
		data = tensor_start_chan(send_tensor, 0, 0 /*R*/);
		for (i = 0; i < send_tensor->height * send_tensor->width; i++)
			data[i] = (data[i] - 0.485)/0.229;
		data = tensor_start_chan(send_tensor, 0, 1 /*G*/);
		for (i = 0; i < send_tensor->height * send_tensor->width; i++)
			data[i] = (data[i] - 0.456)/0.224;
		data = tensor_start_chan(send_tensor, 0, 2 /*B*/);
		for (i = 0; i < send_tensor->height * send_tensor->width; i++)
			data[i] = (data[i] - 0.406)/0.225;
#endif
		recv_tensor = OnnxRPC(socket, send_tensor, VIDEO_RIFE_REQCODE, &rescode);
		if (tensor_valid(recv_tensor)) {
			syslog_info("OK.");
			// SaveTensorAsImage(recv_tensor, input_file);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
		image_destroy(send_image);
	}

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option] <image files>\n", cmd);
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
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = 1; i < argc; i++)
			rife(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
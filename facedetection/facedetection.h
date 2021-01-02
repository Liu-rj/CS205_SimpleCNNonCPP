#include <iostream>

using namespace std;
using namespace cv;

struct conv_param {
	int pad;
	int stride;
	int kernel_size;
	int in_channels;
	int out_channels;
	float* p_weight;
	float* p_bias;
};

struct fc_param {
	int in_features;
	int out_features;
	float* p_weight;
	float* p_bias;
};

float* convertRGB(Mat img);

float* ConvBNReLU(float* img, int rows, int cols, int channels, conv_param& cp);

float* paddling(float* img, int newrows, int newcols, int channels, int pad);

float* im2col(float* newimg, int newrows, int newcols, int convrows, int convcols, int channels, int kernel_size, int stride, int size);

float* MaxPooling(float* img, int convrows, int convcols, int channels);

float* FullConnect(float* img, int rows, int cols, int channels, fc_param& fc);

void SoftMax(float* fcl, int size);

float* cnn(float* img, int rows, int cols, int channels);
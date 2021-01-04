#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetection.h"
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main()
{
	Mat m = imread("./pics/face.jpg");
	Mat image;
	m.convertTo(image, CV_32FC3);
	float* result = nullptr;
	float* img = nullptr;
	try
	{
		img = convertRGB(image);
	}
	catch (const char* e)
	{
		cout << e << endl;
	}
	auto start = std::chrono::steady_clock::now();
	try
	{
		result = cnn(img, m.rows, m.cols, m.channels());
	}
	catch (const char* e)
	{
		cout << e << endl;
		exit(0);
	}
	auto end = std::chrono::steady_clock::now();
	printf("%s%.4f\n", "Confidence score of background: ", result[0]);
	cout << "---------------------------------------" << endl;
	printf("%s%.4f\n", "Confidence score of face: ", result[1]);
	cout << "---------------------------------------" << endl;
	printf("%s%lld%s", "calculation takes ", duration_cast<std::chrono::milliseconds>(end - start).count(), " ms\n");
	imshow("test_img", m);
	waitKey(0);
}
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetection.h"
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main()
{
	Mat m = imread("./pics/kid.jpg");
	Mat image;
	m.convertTo(image, CV_32FC3);
	float* img = convertRGB(image);
	auto start = std::chrono::steady_clock::now();
	float* result = cnn(img, m.rows, m.cols, m.channels());
	auto end = std::chrono::steady_clock::now();
	printf("%s%.4f\n", "Confidence score of background: ", result[0]);
	cout << "---------------------------------------" << endl;
	printf("%s%.4f\n", "Confidence score of face: ", result[1]);
	cout << "---------------------------------------" << endl;
	printf("%s%lld%s", "calculation takes ", duration_cast<std::chrono::milliseconds>(end - start).count(), " ms\n");
	imshow("test_img", m);
	waitKey(0);
}

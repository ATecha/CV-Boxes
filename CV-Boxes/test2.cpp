#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <iostream>
using namespace std;

const int cornerBlockSize = 2;		// These two values seem pretty good.
const int cornerApertureSize = 3;	// ""
const double cornerK = 0.04;		// This one might need some finesse.
const int cornerThreshold = 100;	// Much lower or higher seems to identify too many/too few circles respectively.

//originally 2.0 and 7,7
const double sigma = 50.0; //sigma in x and y for gaussian blur
const Size kSize = Size(11,11); //kernel size for gaussian blur

void displayWindow(string name, Mat& const image) {
	namedWindow(name, WINDOW_KEEPRATIO);
	//resizeWindow("Grey", 500, 500);
	imshow(name, image);
	string filename = name + ".jpg";
	imwrite(filename, image);
	std::cout << "Saved as " << filename << std::endl;
	waitKey(0);
}

Mat cornerDetector(Mat& image) {
	Mat result = Mat::zeros(image.size(), CV_32FC1);

	Mat result_scaled;
	cornerHarris(image, result, cornerBlockSize, cornerApertureSize, cornerK);
	normalize(result, result, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(result, result_scaled);

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			if ((int)result.at<float>(i, j) > cornerThreshold)
			{
				circle(result_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
			}
		}
	}

	return result_scaled;
}


int main(int argc, const char* argv[]) {

	//testing with: cube1.jpg, full-15.jpg, full-72.jpg
	Mat input = imread("full-72.jpg");


	//first convert to greyscale
	Mat grey = input.clone();
	cvtColor(grey, grey, COLOR_BGR2GRAY);
	displayWindow("Grey", grey);

	//then blur to reduce noise (must do before any feature detection)
	Mat blur = grey.clone();
	GaussianBlur(blur, blur, kSize, sigma, sigma, 4);
	displayWindow("Blurred", blur);

	//found this in OpenCV - "calculates a feature map for corner detection"
	//input Mat must be single-channel 8-bit floating-point image
	//output is same type as input
	Mat precorner = blur.clone();
	//converting to greyscale changes it to a single-channel image.
	//just need to convert to 8-bit, floating-point
	precorner.convertTo(precorner, CV_32F); //convert to 32-bit float
	//displayWindow("precorner_converted", precorner);
	preCornerDetect(precorner, precorner, 3, 4); //sobel kernel size 3 works best
	displayWindow("precorner", precorner);


	//use Harris Corner detector
	//Mat corners = blur.clone();
	//cornerDetector(corners);
	//displayWindow("Corners", corners);


	system("pause");
	return 0;

}
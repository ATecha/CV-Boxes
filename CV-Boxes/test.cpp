#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

// Harris Corner Detector Parameters
const int cornerBlockSize = 2;		// These two values seem pretty good.
const int cornerApertureSize = 3;	// ""
const double cornerK = 0.04;		// This one might need some finesse.
const int cornerThreshold = 120;	// Much lower or higher seems to identify too many/too few circles respectively.


// Gaussian Blur Parameters
const double sigma = 0.75;			// sigma in x and y for gaussian blur
const Size kSize = Size(11, 11);	// kernel size for gaussian blur

// Display Window Method - Pops up a window with the image and outputs a file by name.
/////////////////////////////////////////////////////////////////////////////////
void displayWindow(string name, Mat& const image) {
	namedWindow(name, WINDOW_KEEPRATIO);
	//resizeWindow("Grey", 500, 500);
	imshow(name, image);
	string filename = name + ".jpg";
	imwrite(filename, image);
	std::cout << "Saved as " << filename << std::endl;
	waitKey(0);
}

// Canny Edge Detection Method - Returns an image with white edges.
/////////////////////////////////////////////////////////////////////////////////
Mat cannyEdges(Mat& image) {
	Mat result(image);
	
	Canny(result, result, 25, 200, 3);
	return result;
}

// Contour Detection Method - Returns an image with highlighted contours.
/////////////////////////////////////////////////////////////////////////////////
Mat contourDetector(Mat& image) {
	Mat result = Mat::zeros(image.size(), CV_8UC3);
	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));

	
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(result, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	
	return result;
}

// Corner Detector Method - Returns an image with highlighted corners. ** Not Working Well **
/////////////////////////////////////////////////////////////////////////////////
Mat cornerDetector(Mat& image) {
	Mat result = Mat::zeros(image.size(), CV_32FC1);

	Mat result_scaled;
	cornerHarris(image, result, cornerBlockSize, cornerApertureSize, cornerK);
	normalize(result, result, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs( result, result_scaled );

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

// Gaussian Blur Method - Returns a Gaussian Blur of the image.
/////////////////////////////////////////////////////////////////////////////////
Mat gaussianBlur(Mat& image) {
	Mat result(image);

	GaussianBlur(result, result, kSize, sigma, sigma, 4);
	return result;
}

// Grayscale Image Method - Returns a Grayscale Copy of the image.
/////////////////////////////////////////////////////////////////////////////////
Mat grayscaleImage(Mat& image) {
	Mat result(image);

	cvtColor(result, result, COLOR_BGR2GRAY);
	return result;
}

// Process Image Method - Performs a series of OpenCV Actions on the image.
/////////////////////////////////////////////////////////////////////////////////
void processImage(Mat& image) {
	Mat result = grayscaleImage(image);
	displayWindow("grayscale", result);

	// result = gaussianBlur(result);
	// displayWindow("gaussianBlur", result);

	//maybe up the contrast first?
	result.convertTo(result, CV_8U, 1, -100); //increase saturation
	displayWindow("saturated", result);
	result.convertTo(result, -1, 2, 0); //increase contrast
	displayWindow("contrast", result);

	result = cannyEdges(result);
	displayWindow("canny", result);

	result = contourDetector(result);
	displayWindow("contour", result);


	// result = cornerDetector(result);
	// displayWindow("cornerHarris", result);
}

// Main - Takes images as command line arguments, one at a time.
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char* const argv[]) {

	//-- Branch Based on Command Line Arguments --//
	if (argc > 1) {
		for (int index = 1; index < argc; index++) {  // Process each image.
			cout << "Processing '" << argv[index] << "' ..." << endl;

			Mat inputImage = imread(argv[index]);
			if (inputImage.empty()) { // In case of imread failure (invalid filename, format, etc).
				cout << "Unable to process '" << argv[index] << "'" << endl;
			}
			else {
				processImage(inputImage);
				cout << "Processed '" << argv[index] << "' successfully." << endl;
				cout << endl;
			}
		}
	} else { // Process test.jpg if no files are specified.
		cout << "Processing 'test.jpg' ..." << endl;

		Mat inputImage = imread("test.jpg");
		if (inputImage.empty()) {
			cout << "Unable to process 'test.jpg'" << endl;
		} else {
			processImage(inputImage);
			cout << "Processed 'test.jpg' successfully." << endl;
		}
	}

	//-- End Program --//
	return 0;

}
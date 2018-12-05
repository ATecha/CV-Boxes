#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
using namespace std;


// Display Window Method - Pops up a window with the image and outputs a file by name.
/////////////////////////////////////////////////////////////////////////////////
void display(string name, Mat& const image) {
	// namedWindow(name, WINDOW_KEEPRATIO);		// Temporarily commented out for
	// imshow(name, image);						// rapid multi-image processing.

	string filename = name + ".jpg";
	imwrite(filename, image);
	std::cout << "Saved as " << filename << std::endl;

	// waitKey(0);
}


// Grayscale Image Method - Returns a Grayscale Copy of the image.
/////////////////////////////////////////////////////////////////////////////////
Mat grayscale(Mat& image) {
	Mat result = image.clone();

	cvtColor(result, result, COLOR_BGR2GRAY);
	result.convertTo(result, CV_8U, 1, 0);
	return result;
}


// Gaussian Blur Method - Returns a Gaussian Blur copy of the image.
// Count - Numer of times to repeat the blur.
// Sigma - Gaussian kernel standard deviation (in X/Y).
// kSize - Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd.
//		   Or, they can be zero's and then they are computed from sigma.
// BorderType - Pixel extrapolation method. Default 4. Probably don't need to change this.
/////////////////////////////////////////////////////////////////////////////////
Mat gaussian(Mat& image, int count = 1, double sigma = 1.0, Size kSize = Size(3, 3), int borderType = 4) {
	Mat result = image.clone();

	for (int i = 0; i < count; i++) {
		GaussianBlur(result, result, kSize, sigma, sigma, borderType);
	}
	return result;
}


// Median Blur Method - Returns a Median Blur copy of the image.
// Count - Numer of times to repeat the blur.
// kSize - Aperture linear size; it must be odd and greater than 1.
/////////////////////////////////////////////////////////////////////////////////
Mat median(Mat& image, int count = 1, int kSize = 3) {
	Mat result = image.clone();

	for (int i = 0; i < count; i++) {
		medianBlur(result, result, kSize);
	}
	return result;
}


// Bilateral Filtering Method - Returns a Bilateral Filtered copy of the image.
// Count - Numer of times to repeat the blur.
// Diameter - Diameter of each pixel neighborhood that is used during filtering.
// Sigma - Filter strength.
// BorderType - Pixel extrapolation method. Default 4. Probably don't need to change this.
/////////////////////////////////////////////////////////////////////////////////
Mat bilat(Mat& image, int count = 1, int diameter = 3, double sigma = 50.0) {
	Mat result = image.clone();

	for (int i = 0; i < count; i++) {
		bilateralFilter(image, result, diameter, sigma, sigma);
	}
	return result;
}


// Desaturate Method - Returns a (de)saturated copy of the image.
// Value - Saturation intensity (can be negative).
/////////////////////////////////////////////////////////////////////////////////
Mat saturate(Mat& image, int value = 100) {
	Mat result = image.clone();

	result.convertTo(result, CV_8U, 1, value);
	return result;
}


// Contrast Method - Returns a copy of the image with adjusted contrast.
// Parameters:
// Value - Contrast intensity (can be negative).
/////////////////////////////////////////////////////////////////////////////////
Mat contrast(Mat& image, int value = 2) {
	Mat result = image.clone();

	result.convertTo(result, CV_8U, value, 0);
	return result;
}


// Canny Edge Detection Method - Returns an image with Canny edges.
// Parameters:
// Threshold 1 - First threshold for the hysteresis procedure.
// Threshold 2 - Second threshold for the hysteresis procedure.
// ApertureSize - Aperture size for the Sobel operator.
/////////////////////////////////////////////////////////////////////////////////
Mat canny(Mat& image, double threshold1 = 75, double threshold2 = 110, int apertureSize = 3) {
	Mat result = image.clone();
	
	Canny(result, result, threshold1, threshold2, apertureSize);
	return result;
}

// Cull Gray Method - Helper for Laplacian, removes gray values under a given intensity.
// Parameters:
// Threshold - Value below which values are changed to black.
void cullGray(Mat& image, int threshold) {
	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++) {	// If average color intensity is below threshold...
			if (image.at<uchar>(r, c) < threshold) {
				image.at<uchar>(r, c) = 0;	// Paint it Black
			}
		}
	}
}

// Laplacian Method - Returns an image with Laplacian edges.
// Parameters:
// Depth - Desired depth of the destination image.
// kSize - Aperture size used to compute the second-derivative filters. The size must be positive and odd.
/////////////////////////////////////////////////////////////////////////////////
Mat laplacian(Mat& image, int kSize = 3) {
	Mat result = image.clone();

	Laplacian(result, result, CV_8U, kSize);

	// bilat(result, 3);
	cullGray(result, 150);

	return result;
}


// Hough Transform Method - Returns an image with hough lines.
// Parameters:
// Rho - Distance resolution of the accumulator, in pixels.
// Theta - Angle resolution of the accumulator, in radians.
// Threshold - Only lines with at least this many votes get returned.
/////////////////////////////////////////////////////////////////////////////////
Mat hough(Mat& image, double rho = 1.0, double theta = 3, int threshold = 150) {
	theta = theta * CV_PI / 180; // ( theta * pi/180 where theta is degrees )

	Mat result;
	cvtColor(image, result, COLOR_GRAY2BGR);

	vector<Vec2f> lines;
	HoughLines(image, lines, rho, theta, threshold, 0, 0); // in, lines out, rho, theta, threshold, srn, stn, min theta, max theta

	//from https://docs.opencv.org/master/d5/df9/samples_2cpp_2tutorial_code_2ImgTrans_2houghlines_8cpp-example.html#a8
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(result, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	
	return result;
}


// Helper Function From: https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp
// Finds a cosine of angle between vectors.
// From pt0->pt1 and From pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}


// Contour Detection Method - Returns an image with highlighted contours.
/////////////////////////////////////////////////////////////////////////////////
Mat contour(Mat& image) {
	Mat result = Mat::zeros(image.size(), CV_8UC3);
	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0)); // CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_KCOS CHAIN_APPROX_TC89_L1

	vector<Point> approx;
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(result, contours, i, color, 2, 8, hierarchy, 0, Point());

		//// From: https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp
		///////
		//approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

		//if (approx.size() == 4 &&
		//	fabs(contourArea(approx)) > 1000 &&
		//	isContourConvex(approx))
		//{
		//	double maxCosine = 0;

		//	for (int j = 2; j < 5; j++)
		//	{
		//		// find the maximum cosine of the angle between joint edges
		//		double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
		//		maxCosine = MAX(maxCosine, cosine);
		//	}

		//	// if cosines of all angles are small
		//	// (all angles are ~90 degree) then write quandrange
		//	// vertices to resultant sequence
		//	if (maxCosine < 0.3) {
		//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//		drawContours(result, contours, i, color, 2, 8, hierarchy, 0, Point());
		//	}
		//}
		
	}
	
	return result;
}


// Corner Detector Method - Returns an image with highlighted corners.
// Parameters:
// BlockSize - Neighborhood size.
// ApertureSize - Aperture parameter for the Sobel operator.
// K - Harris detector free parameter.
// Threshold - Only circles corners with an intensity above this value.
/////////////////////////////////////////////////////////////////////////////////
Mat corner(Mat& image, int blockSize = 2, int apertureSize = 3, double k = 0.04, int threshold = 120) {

	Mat result = Mat::zeros(image.size(), CV_32FC1);

	Mat result_scaled;
	cornerHarris(image, result, blockSize, apertureSize, k);
	normalize(result, result, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs( result, result_scaled );

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			if ((int)result.at<float>(i, j) > threshold)
			{
				circle(result_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
			}
		}
	}

	return result_scaled;
}


// Process Image Method - Performs a series of OpenCV Actions on the image.
//					****************************************
//					*** This is where the magic happens! ***
//					****************************************
/////////////////////////////////////////////////////////////////////////////////
void processImage(Mat& image, string filename) {
	Mat result = grayscale(image);
	// display(filename + "_gray", result);

	// Saturation/Contrast Pass
	result = saturate(result, -50);		// Decrease Saturation	- Can pass in second parameter as saturation amount.
	result = contrast(result, 1.5);		// Increase Contrast	- Can pass in second parameter as contrast amount.
	display(filename + "_satcon1", result);

	// Median Blur
	result = median(result, 3, 3);
	display(filename + "_median", result);

	// Bilat Blur
	// result = bilat(result, 8, 4, 150);
	// display(filename + "_bilat", result);

	// Gaussian Blur
	// result = gaussian(result, 5, 2.0, Size(3,3));
	// display(filename + "_gauss", result);

	// Saturation/Contrast Pass
	// result = saturate(result, -100);	// Decrease Saturation
	// result = contrast(result, 2);		// Increase Contrast
	// display(filename + "_satcon2", result);

	// Get Canny Edges
	// Mat cannyImg = canny(result, 50, 80);
	// display(filename + "_canny", cannyImg);

	// Get Laplacian Edges
	Mat laplacianImg = laplacian(result, 5);
	display(filename + "_laplacian", laplacianImg);

	// Get Contours from Canny Edges
	Mat contourImg = contour(laplacianImg);
	display(filename + "_contours", contourImg);

	// Get Hough Lines from Canny Edges
	Mat lineImg = hough(laplacianImg);
	display(filename + "_lines", lineImg);

	// Get Corners from Canny Edges
	Mat cornerImg = corner(laplacianImg);
	display(filename + "_corners", cornerImg);
}


// Main - Takes images as command line arguments, one at a time.
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char* const argv[]) {

	//-- Branch Based on Command Line Arguments --//
	if (argc > 1) {
		for (int index = 1; index < argc; index++) {  // Process each image.
			cout << "Processing '" << argv[index] << "' ..." << endl;
			string shortFilename = argv[index];
			size_t pos = shortFilename.find('.');
			shortFilename = shortFilename.substr(0, pos);

			Mat inputImage = imread(argv[index]);
			if (inputImage.empty()) { // In case of imread failure (invalid filename, format, etc).
				cout << "Unable to process '" << argv[index] << "'" << endl;
			}
			else {
				processImage(inputImage, shortFilename);
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
			processImage(inputImage, "test");
			cout << "Processed 'test.jpg' successfully." << endl;
		}
	}

	//-- End Program --//
	return 0;

}
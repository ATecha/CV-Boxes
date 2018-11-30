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

Mat grayscaleImage(Mat& image) {
	Mat result(image);

	cvtColor(result, result, COLOR_BGR2GRAY);
	return result;
}

void processImage(Mat& image) {
	Mat result = grayscaleImage(image);
	imwrite("grayscale.jpg", result);

	result = cornerDetector(result);
	imwrite("cornerHarris.jpg", result);
}

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
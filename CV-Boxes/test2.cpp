#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> // might need?
using namespace cv;

#include <iostream>
using namespace std;

const int cornerBlockSize = 2;		// These two values seem pretty good.
const int cornerApertureSize = 3;	// ""
const double cornerK = 0.04;		// This one might need some finesse.
const int cornerThreshold = 100;	// Much lower or higher seems to identify too many/too few circles respectively.

//originally 2.0 and 7,7
const double sigma = 0.75; //sigma in x and y for gaussian blur
const Size kSize = Size(11,11); //kernel size for gaussian blur

//PURPOSE: display image in GUI and save to disk as a .jpg
//PRE: input image exists
//POST: new GUI window is display image then saves image to disk.
//The same name is used for the GUI window as for the saved file.
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


//PURPOSE: main method
//PRE: image is in the same location as this program
//IMAGE PRECONDITIONS: (all conditions we require for the box counting to work, such as angle, perspective, etc)
//POST: 
int main2(int argc, const char* argv[]) {

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

	//let's see if the saved image is different from what's in memory...
	//Mat precorner_saved = imread("precorner.jpg"); //comment back in to test
	//displayWindow("saved", precorner_saved);
	//nope, it's definitely different.

	//use Harris Corner detector
	//Mat corners = precorner_saved.clone(); //comment back in to test
	//cornerDetector(corners);
	//displayWindow("Corners", corners);

	//using what's in memory
	/*
	precorner.convertTo(precorner, CV_32F);
	cornerHarris(precorner, precorner, 2, 3, 0.04); //in, out, blockSize, kSize, k free param
	displayWindow("HarrisCorner_mem", precorner);
	*/

	//or try Harris corner detection on just the blurred image, without the precorner detection
	Mat harris = blur.clone();
	harris.convertTo(harris, CV_32F);
	cornerHarris(harris, harris, 2, 3, 0.04);
	//displayWindow("Harris_nopre", harris);	

	//cornerDetector(harris);
	//displayWindow("Harris_nopre_custom", harris);


	/* not currently working...
	//using the saved-to-disk output from precorners operation
	corners.convertTo(corners, CV_32F);
	cornerHarris(corners, corners, 2, 3, 0.04);
	displayWindow("HarrisCorner_saved", corners);
	*/

	//try edge detect on the precorner image
	Mat edges = imread("precorner.jpg", IMREAD_GRAYSCALE);
	//note OpenCV Canny edge detector blurs, finds intensity gradient, does non-maxima surpression, and hysteresis thresholding
	//all in the canny method. No need to do it seperately (though I guess it can't hurt to have it pre-blurred...)
	//edges.convertTo(edges, CV_8UC1);
	Canny(edges, edges, 100, 300, 3); //in, out, low threshold, high threshold, kernel size 3-7
	displayWindow("Edges", edges);
	
	/*
	//code for line segment detection borrowed from:
	//https://docs.opencv.org/3.4/d3/dff/samples_2cpp_2lsd_lines_8cpp-example.html#a9
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	vector<Vec4f> lines_std;
	ls->detect(edges, lines_std);

	ls->drawSegments(edges, lines_std);
	displayWindow("LineSegs", edges);*/

	//the above doesn't give very good results
	//so what if we adjust the blur? Let's start over...
	Mat image = imread("full-72.jpg", IMREAD_GRAYSCALE); //read in in greyscale
	Size size = Size(5, 5);
	double sigma2 = 2.0;
	GaussianBlur(image, image, size, sigma2, sigma2);
	Canny(image, image, 50, 75, 3);
	displayWindow("Edges2", image);
	//code for line segment detection borrowed from:
//https://docs.opencv.org/3.4/d3/dff/samples_2cpp_2lsd_lines_8cpp-example.html#a9
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	vector<Vec4f> lines_std;
	ls->detect(image, lines_std);
	ls->drawSegments(image, lines_std);
	displayWindow("LineSegs2", image);
	//the above block does somewhat better.
	//either way, we have the lines segments saved in the ls vector, so we can do something with them.
	//I'm not sure how line segments work in OpenCV, though. Something to look more into.

	//Another thing to try would be HoughLinesP
	//as suggested in the process here: https://stackoverflow.com/questions/31772804/opencv-detect-cubes-corners




	//doing it without the blur
	//Canny(grey, grey, 50, 200, 3);
	//displayWindow("untreated", grey);


	system("pause");
	return 0;

}
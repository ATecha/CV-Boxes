#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp> // might need?
using namespace cv;

#include <iostream>
using namespace std;

//PURPOSE: display image in GUI and save to disk as a .jpg
//PRE: input image exists
//POST: new GUI window is display image then saves image to disk.
//The same name is used for the GUI window as for the saved file.
void display(string name, Mat& const image) {
	namedWindow(name, WINDOW_KEEPRATIO);
	//resizeWindow("Grey", 500, 500);
	imshow(name, image);
	string filename = name + ".jpg";
	imwrite(filename, image);
	std::cout << "Saved as " << filename << std::endl;
	waitKey(0);
}

int main(int argc, const char* argv[]) {

	//testing with: cube1.jpg, full-15.jpg, full-72.jpg
	Mat input = imread("full-72.jpg");

	Mat grey = input.clone();
	cvtColor(grey, grey, COLOR_BGR2GRAY);

	//maybe up the contrast first?
	grey.convertTo(grey, CV_8U, 1, -100); //increase saturation
	display("saturated", grey);
	grey.convertTo(grey, -1, 2, 0); //increase contrast
	display("contrast", grey);

	//Canny(grey, grey, 25, 100, 3);
	Canny(grey, grey, 25, 200, 3);
	display("canny2", grey);

	Mat color_out;
	cvtColor(grey, color_out, COLOR_GRAY2BGR);

	vector<Vec2f> lines;
	HoughLines(grey, lines, 1, CV_PI / 180, 150, 0, 0); //in, lines out, rho, theta, threshold, srn, stn, min theta, max theta

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
		line(color_out, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	display("hough2", color_out);


	system("pause");
	return 0;
}

int mainB(int argc, const char* argv[]) {

	//testing with: cube1.jpg, full-15.jpg, full-72.jpg
	Mat input = imread("full-72.jpg");

	//Mat grey = input.clone();
	Mat grey = input.clone();
	cvtColor(grey, grey, COLOR_BGR2GRAY);
	display("Grey", grey);

	Mat canny = grey.clone();
	Canny(canny, canny, 50, 200, 3); //in, out, low threshold, high threshold, kernel size 3-7
	display("Canny", canny);

	//Mat grey = input.clone();
	//Mat grey = canny.clone();
	//cvtColor(grey, grey, COLOR_BGR2GRAY);
	//display("Grey", grey);

	//make a black and white (binary) copy of the input image
	//Mat bw = grey.clone();
	Mat bw = canny.clone();
	threshold(bw, bw, 100, 255, THRESH_BINARY); //convert it to a binary image
	display("Binary", bw);

	//from example here
	//http://opencvexamples.blogspot.com/2013/10/line-detection-by-hough-line-transform.html
	//should go: read in, greyscale, canny, Hough lines, draw lines
	double rho = 1.0; //distance resolution of the accumulator, in pixels
	double theta = CV_PI/180; //angle resolution of the accumulator, in radians
	int threshold  = 150; //accumulator threshold param - only lines with at least this many votes get returned
	vector<Vec2f>lines; //array of lines found. We want to save them so we can draw or compare them!
	HoughLines(bw, lines, rho, theta, threshold);


	Mat hough = bw.clone();
	cvtColor(hough, hough, COLOR_GRAY2BGR); //convert back to color, so colored lines will show up
	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float r = lines[i][0], t = lines[i][1];
		Point pt1, pt2;
		double a = cos(t), b = sin(t);
		double x0 = a * r, y0 = b * t;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(hough, pt1, pt2, Scalar(0, 0, 255), 3);
	}

	display("hough", hough);

	system("pause");
	return 0;
}
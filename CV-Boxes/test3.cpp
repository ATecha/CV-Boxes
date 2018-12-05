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
	grey.convertTo(grey, CV_8U, 1, -75); //decrease saturation
	//display("saturated", grey);
	grey.convertTo(grey, CV_8U, 2, 0); //increase contrast
	//display("contrast", grey);

	//second pass
	grey.convertTo(grey, CV_8U, 1, -100); //decrease saturation
	//display("saturated2", grey);
	grey.convertTo(grey, CV_8U, 2, 0); //increase contrast
	//display("contrast2", grey);

	//try blurring?
	//blurring helps, since adjusting the saturation and contrast magnifies noise
	double sigma = 1.0;
	GaussianBlur(grey, grey, Size(3,3), sigma, sigma, 4);
	//display("blur", grey);

	//Canny(grey, grey, 25, 100, 3);
	Canny(grey, grey, 75, 110, 3);
	display("canny2", grey);

	Mat color_out;
	cvtColor(grey, color_out, COLOR_GRAY2BGR);

	vector<Vec2f> lines;
	double rho = 1.0; //distance resolution of the accumulator, in pixels
	double theta = 3 * CV_PI / 180; //angle resolution of the accumulator, in radians ( n * pi/180 where n is degrees)
	int threshold = 150; //accumulator threshold param - only lines with at least this many votes get returned
	HoughLines(grey, lines, rho, theta, threshold, 0, 0); //in, lines out, rho, theta, threshold, srn, stn, min theta, max theta

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
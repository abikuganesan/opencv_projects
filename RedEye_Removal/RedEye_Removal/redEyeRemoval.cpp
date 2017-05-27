
/*
 * Red Eye Removal Adapted From Satya Mallick
 */

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void fillHoles(Mat &mask)
{
	Mat maskFloodfill = mask.clone();
	floodFill(maskFloodfill, cv::Point(0, 0), Scalar(255));
	Mat mask2;
	bitwise_not(maskFloodfill, mask2);
	mask = (mask2 | mask);
}

int main()
{
	Mat orgIMG = imread("Test Images/red_eyes.jpg", CV_LOAD_IMAGE_COLOR);
	Mat dstIMG = orgIMG.clone();

	// Detect Eyes
	CascadeClassifier eyesCascade("haarcascade_eye.xml");
	vector<Rect> eyes;
	eyesCascade.detectMultiScale(orgIMG, eyes, 1.3, 4, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));

	// For Each Eye
	for (size_t i = 0; i < eyes.size(); i++)
	{
		Mat eye = orgIMG(eyes[i]);
		vector<Mat>bgr(3);
		split(eye, bgr);

		// Red Eye Detector 
		Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));

		fillHoles(mask);
		dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

		Mat mean = (bgr[0] + bgr[1]) / 2;
		mean.copyTo(bgr[2], mask);
		mean.copyTo(bgr[0], mask);
		mean.copyTo(bgr[1], mask);

		// Merge BGR Channels
		Mat eyeOut;
		cv::merge(bgr, eyeOut);

		// Fixed Eye to Image
		eyeOut.copyTo(dstIMG(eyes[i]));
	}
	
	// Result
	imshow("Original Image", orgIMG);
	imshow("Edited Image", dstIMG);
	waitKey(0);

	return 0;
}

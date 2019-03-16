#include "Visualize.h"

using namespace cv;
using namespace std;

Visualize::Visualize(Mat &image, int confidence, vector<DetectedObject> &results) : mImage(image), mConfidence(confidence), mResults(results) {

	if (mImage.empty()) {
		printf("Image not found\n");
		exit(1);
	}
};

Visualize::~Visualize() {};

void Visualize::show() {
	Mat img_cp = mImage.clone();

	resize(img_cp, img_cp, Size(mWidth, mHeight));
	float xratio = (float)mImage.cols / (float)mWidth;
	float yratio = (float)mImage.rows / (float)mHeight;
	int detectedNum = (int)mResults.size();
	for (int i = 0; i < detectedNum; i++) {
		
		float confidence = mResults[i].confidence;
		confidence = confidence * 100;

		if (confidence > mConfidence) {

			float left = mResults[i].left * xratio;
			float top = mResults[i].top * yratio;
			float right = (mResults[i].right - mResults[i].left) * xratio + left;
			float bottom = (mResults[i].bottom - mResults[i].top) * yratio + top;
			int index = mResults[i].objType % mColorNum;
			Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
			string txt = mResults[i].name;
			rectangle(mImage, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
			Size size = getTextSize(txt, FONT_HERSHEY_SIMPLEX, 0.6, 1, 0);
			int width = size.width;
			int height = size.height;
			rectangle(mImage, Point((int)left, ((int)bottom - 5) - (height + 5)), Point(((int)left + width), ((int)bottom - 5)), clr, -1);
			putText(mImage, txt, Point(((int)left + 5), ((int)bottom - 10)), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, 8);
		}
	}
	resize(img_cp, img_cp, Size(mImage.cols, mImage.rows));
	imshow("Detected Image", mImage);

}

void Visualize::LegendImage() {
	string window_name = "AMD Object Detection - Legend";

	Size legendGeometry = Size(325, (20 * 40) + 40);
	Mat legend = Mat::zeros(legendGeometry, CV_8UC3);
	Rect roi = Rect(0, 0, 325, (20 * 40) + 40);
	legend(roi).setTo(Scalar(255, 255, 255));

	for (int l = 0; l < 20; l++) {
		Scalar clr(colors[l][0], colors[l][1], colors[l][2]);
		string className = yoloClasses[l];
		putText(legend, className, Point(20, (l * 40) + 30), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1, 8);
		rectangle(legend, Point(225, (l * 40)), Point(300, (l * 40) + 40), clr, -1);
	}
	imshow(window_name, legend);

	return;
}


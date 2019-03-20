#pragma once

#include "Common.h"
#include "vx_ext_opencv.h"

static const int colors[20][3] = {
			{160,82,45},        // aeroplane
			{128,0,0},          // bicycle
			{47,79,79},         // bird
			{155,140,145},      // boat
			{140,155,255},      // bottle
			{255,105,180},      // bus
			{255,0,0},          // car
			{75,0,130},         // cat
			{255,140,0},        // chair
			{250,128,114},      // cow
			{153,50,204},       // diningtable
			{130,230,150},      // dog
			{0,220,255},        // horse
			{0,191,255},        // motorbike
			{0,0,255},          // person
			{0,255,255},        // potted plant
			{107,142,35},       // sheep
			{0,128,0},          // sofa
			{124,252,0},        // train
			{199,21,133}        // tvmonitor
};

static const std::string yoloClasses[20] = {
			"aeroplane",
			"bicycle",
			"bird",
			"boat",
			"bottle",
			"bus",
			"car",
			"cat",
			"chair",
			"cow",
			"dining table",
			"dog",
			"horse",
			"motorbike",
			"person",
			"potted plant",
			"sheep",
			"sofa",
			"train",
			"tvmonitor"
};

class Visualize {
public:
	Visualize(cv::Mat &image, int confidence, std::vector<DetectedObject> &results);
	~Visualize();
	void show();
	void LegendImage();

private:
	const int mConfidence;
	const int mColorNum = 20;
	const int mWidth = 416;
	const int mHeight = 416;
	cv::Mat &mImage;
	std::vector<DetectedObject> mResults;

};

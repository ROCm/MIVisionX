#pragma once

#include "Region.h"
#include "Visualize.h"
#include "vx_ext_winml.h"
#include "vx_ext_opencv.h"
#include "vx_ext_amd.h"

class AnnieYoloDetect {
public:
	//constructor
	AnnieYoloDetect(std::string mInput, std::string modelLoc, int confidence, int mode);
	
	//destructor
	~AnnieYoloDetect();

	//run inference on given input and show the detected bounding boxes
	void detect();

private:
	const int mMode;
	const int mConfidence;
	const int mWidth = 416;
	const int mHeight = 416;
	const std::string mInput;
	const std::string mModelLoc;
	std::unique_ptr<Region> mRegion;
	std::unique_ptr<Visualize> mVisualize;
};
#include "MSERDetector.h"

MSERDetector::MSERDetector(const int minSize){
    ms = cv::MSER::create(5, minSize);
}

MSERDetector::~MSERDetector(){
}

void MSERDetector::detect(const cv::Mat &img, std::vector<std::vector<cv::Point>> &regions, std::vector<cv::Rect> &mser_bbox) {
    ms->detectRegions(img, regions, mser_bbox);
}
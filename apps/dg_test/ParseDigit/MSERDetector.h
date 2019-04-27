#pragma once

#include <opencv2/opencv.hpp>

class MSERDetector {

public:
    MSERDetector(const int minSize);
    ~MSERDetector();
    
    void detect(const cv::Mat &img, std::vector<std::vector<cv::Point>> &regions, std::vector<cv::Rect> &mser_bbox);

private:
    cv::Ptr<cv::MSER> ms;
};
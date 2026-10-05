// opencv_remap_bench.cpp
// Build: g++ -O2 -std=c++17 -I/usr/include/opencv4 opencv_remap_bench.cpp -lopencv_core -lopencv_imgproc -o opencv_remap_bench
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static double median(std::vector<double> &v) {
    size_t n = v.size();
    std::sort(v.begin(), v.end());
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <width> <height> <u8|rgb|rgbx> [border:0|1] [iterations]\n", argv[0]);
        return 1;
    }
    int W = atoi(argv[1]);
    int H = atoi(argv[2]);
    const char *fmt = argv[3];
    int borderConstant = (argc > 4) ? atoi(argv[4]) : 0;
    int iterations = (argc > 5) ? atoi(argv[5]) : 100;

    cv::Mat src, mapx, mapy;
    int cvType = CV_8UC1;
    if (strcmp(fmt, "rgb") == 0) cvType = CV_8UC3;
    if (strcmp(fmt, "rgbx") == 0) cvType = CV_8UC4;

    src.create(H, W, cvType);
    if (cvType == CV_8UC1) src.setTo(cv::Scalar(0xAA));
    else if (cvType == CV_8UC3) src.setTo(cv::Scalar(0xCC, 0xBB, 0xAA));
    else src.setTo(cv::Scalar(0xCC, 0xBB, 0xAA, 0xAA));

    mapx.create(H, W, CV_32FC1);
    mapy.create(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            mapx.at<float>(y, x) = (float)(W - 1 - x);
            mapy.at<float>(y, x) = (float)(H - 1 - y);
        }
    }

    int borderMode = borderConstant ? cv::BORDER_CONSTANT : cv::BORDER_REPLICATE;
    cv::Scalar borderVal(0, 0, 0, 0);

    for (int i = 0; i < std::max(1, iterations / 10); ++i) {
        cv::Mat tmp;
        cv::remap(src, tmp, mapx, mapy, cv::INTER_LINEAR, borderMode, borderVal);
    }

    std::vector<double> times;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            cv::Mat tmp;
            cv::remap(src, tmp, mapx, mapy, cv::INTER_LINEAR, borderMode, borderVal);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        times.push_back(ms / iterations);
    }
    double med = median(times);
    printf("%d,%d,%s,%s,%.4f\n", W, H, fmt, borderConstant ? "constant" : "replicate", med);
    return 0;
}

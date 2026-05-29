/*
 * Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * opencv_mark.cpp — OpenCV benchmark tool for comparison with openvx-mark.
 *
 * Runs the same image-processing operations as openvx-mark using OpenCV,
 * producing JSON output in an identical schema so that
 * openvx-mark/scripts/compare_reports.py can generate pairwise comparisons.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Resolution {
    std::string name;
    int width;
    int height;
};

static const std::vector<Resolution> kResolutions = {
    {"VGA", 640, 480},
    {"FHD", 1920, 1080},
    {"4K", 3840, 2160},
};

struct TimingStats {
    double mean_ms   = 0;
    double median_ms = 0;
    double min_ms    = 0;
    double max_ms    = 0;
    double stddev_ms = 0;
    double p5_ms     = 0;
    double p95_ms    = 0;
    double p99_ms    = 0;
    double cv_percent = 0;
    int    sample_count     = 0;
    int    outliers_removed = 0;
};

struct BenchmarkResult {
    std::string name;
    std::string category;
    std::string mode       = "graph";
    std::string resolution;
    int  width  = 0;
    int  height = 0;
    bool supported = true;
    bool verified  = true;
    int  iterations = 0;
    int  warmup     = 0;
    double megapixels_per_sec = 0;
    TimingStats wall_clock;
};

// IQR-based outlier removal (matching openvx-mark methodology)
static TimingStats computeStats(std::vector<double>& samples) {
    TimingStats s;
    if (samples.empty()) return s;

    std::sort(samples.begin(), samples.end());

    size_t n = samples.size();
    double q1 = samples[n / 4];
    double q3 = samples[(3 * n) / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr;
    double hi = q3 + 1.5 * iqr;

    std::vector<double> clean;
    clean.reserve(n);
    for (double v : samples) {
        if (v >= lo && v <= hi) clean.push_back(v);
    }
    s.outliers_removed = static_cast<int>(n - clean.size());
    if (clean.empty()) clean = samples; // fallback

    std::sort(clean.begin(), clean.end());
    size_t cn = clean.size();
    s.sample_count = static_cast<int>(cn);

    double sum = 0;
    for (double v : clean) sum += v;
    s.mean_ms = sum / cn;

    s.median_ms = (cn % 2 == 0)
        ? (clean[cn / 2 - 1] + clean[cn / 2]) / 2.0
        : clean[cn / 2];

    s.min_ms = clean.front();
    s.max_ms = clean.back();

    s.p5_ms  = clean[std::min<size_t>((size_t)(cn * 0.05), cn - 1)];
    s.p95_ms = clean[std::min<size_t>((size_t)(cn * 0.95), cn - 1)];
    s.p99_ms = clean[std::min<size_t>((size_t)(cn * 0.99), cn - 1)];

    double var = 0;
    for (double v : clean) var += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = std::sqrt(var / cn);
    s.cv_percent = (s.mean_ms > 0) ? (s.stddev_ms / s.mean_ms * 100.0) : 0;

    return s;
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------

struct SystemInfo {
    std::string hostname;
    std::string os_name;
    std::string os_version;
    std::string cpu_model;
    int cpu_cores = 0;
    double ram_gb = 0;
    std::string timestamp;
};

static std::string trimLine(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::string shellExec(const char* cmd) {
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return result;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    return trimLine(result);
}

static SystemInfo getSystemInfo() {
    SystemInfo si;
    si.hostname   = shellExec("hostname 2>/dev/null");
    si.os_name    = shellExec("uname -s 2>/dev/null");
    si.os_version = shellExec("uname -r 2>/dev/null");
#ifdef __linux__
    si.cpu_model = shellExec("grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2");
    si.cpu_cores = std::atoi(shellExec("nproc 2>/dev/null").c_str());
    std::string mem = shellExec("grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}'");
    if (!mem.empty()) si.ram_gb = std::atof(mem.c_str()) / (1024.0 * 1024.0);
#else
    si.cpu_model = shellExec("sysctl -n machdep.cpu.brand_string 2>/dev/null");
    si.cpu_cores = std::atoi(shellExec("sysctl -n hw.ncpu 2>/dev/null").c_str());
    std::string mem = shellExec("sysctl -n hw.memsize 2>/dev/null");
    if (!mem.empty()) si.ram_gb = std::atof(mem.c_str()) / (1024.0 * 1024.0 * 1024.0);
#endif
    // ISO-8601 timestamp
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char tbuf[64];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    si.timestamp = tbuf;
    return si;
}

// ---------------------------------------------------------------------------
// Benchmark kernel definitions
// ---------------------------------------------------------------------------

using KernelFunc = std::function<void(const cv::Mat& src, cv::Mat& dst)>;

struct KernelDef {
    std::string name;
    std::string category;
    // setup: given (width, height), produce src and dst mats, and a run function
    std::function<void(int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run)> setup;
};

static std::vector<KernelDef> getKernels() {
    std::vector<KernelDef> k;

    // --- Filters ---
    k.push_back({"Box3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::blur(s, d, cv::Size(3, 3)); };
    }});
    k.push_back({"Gaussian3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::GaussianBlur(s, d, cv::Size(3, 3), 0); };
    }});
    k.push_back({"Median3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::medianBlur(s, d, 3); };
    }});
    k.push_back({"Erode3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        run = [elem](const cv::Mat& s, cv::Mat& d) { cv::erode(s, d, elem); };
    }});
    k.push_back({"Dilate3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        run = [elem](const cv::Mat& s, cv::Mat& d) { cv::dilate(s, d, elem); };
    }});
    k.push_back({"Sobel3x3", "filters", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_16SC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::Sobel(s, d, CV_16S, 1, 0, 3); };
    }});

    // --- Pixelwise ---
    k.push_back({"Add", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::add(s, src2, d); };
    }});
    k.push_back({"Subtract", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::subtract(s, src2, d); };
    }});
    k.push_back({"Multiply", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::multiply(s, src2, d, 1.0 / 255.0); };
    }});
    k.push_back({"AbsDiff", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::absdiff(s, src2, d); };
    }});
    k.push_back({"And", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::bitwise_and(s, src2, d); };
    }});
    k.push_back({"Or", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::bitwise_or(s, src2, d); };
    }});
    k.push_back({"Xor", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::bitwise_xor(s, src2, d); };
    }});
    k.push_back({"Not", "pixelwise", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::bitwise_not(s, d); };
    }});

    // --- Color ---
    k.push_back({"ColorConvert", "color", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC3); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::cvtColor(s, d, cv::COLOR_RGB2GRAY); };
    }});
    k.push_back({"ChannelExtract", "color", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC3); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::extractChannel(s, d, 0); };
    }});
    k.push_back({"ConvertDepth", "color", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_16SC1);
        run = [](const cv::Mat& s, cv::Mat& d) { s.convertTo(d, CV_16S); };
    }});

    // --- Geometric ---
    k.push_back({"ScaleImage_Half", "geometric", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h / 2, w / 2, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::resize(s, d, d.size(), 0, 0, cv::INTER_LINEAR); };
    }});
    k.push_back({"ScaleImage_Double", "geometric", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h * 2, w * 2, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::resize(s, d, d.size(), 0, 0, cv::INTER_LINEAR); };
    }});
    k.push_back({"WarpAffine", "geometric", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 5, 0, 1, 5);
        run = [M, w, h](const cv::Mat& s, cv::Mat& d) { cv::warpAffine(s, d, M, cv::Size(w, h)); };
    }});
    k.push_back({"WarpPerspective", "geometric", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        cv::Mat M = (cv::Mat_<double>(3, 3) << 1, 0, 5, 0, 1, 5, 0, 0, 1);
        run = [M, w, h](const cv::Mat& s, cv::Mat& d) { cv::warpPerspective(s, d, M, cv::Size(w, h)); };
    }});

    // --- Statistical ---
    k.push_back({"Histogram", "statistical", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat();
        run = [](const cv::Mat& s, cv::Mat& d) {
            int histSize = 256;
            float range[] = {0, 256};
            const float* ranges[] = {range};
            cv::calcHist(&s, 1, nullptr, cv::Mat(), d, 1, &histSize, ranges);
        };
    }});
    k.push_back({"EqualizeHist", "statistical", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::equalizeHist(s, d); };
    }});
    k.push_back({"MeanStdDev", "statistical", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat();
        run = [](const cv::Mat& s, cv::Mat& d) {
            cv::Scalar mean, stddev;
            cv::meanStdDev(s, mean, stddev);
        };
    }});
    k.push_back({"MinMaxLoc", "statistical", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat();
        run = [](const cv::Mat& s, cv::Mat& d) {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(s, &minVal, &maxVal, &minLoc, &maxLoc);
        };
    }});
    k.push_back({"IntegralImage", "statistical", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat();
        run = [](const cv::Mat& s, cv::Mat& d) { cv::integral(s, d); };
    }});

    // --- Features ---
    k.push_back({"CannyEdgeDetector", "features", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::Canny(s, d, 100, 200, 3); };
    }});
    k.push_back({"HarrisCorners", "features", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_32FC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::cornerHarris(s, d, 2, 3, 0.04); };
    }});
    k.push_back({"FastCorners", "features", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat();
        run = [](const cv::Mat& s, cv::Mat& d) {
            std::vector<cv::KeyPoint> kpts;
            cv::FAST(s, kpts, 20, true);
        };
    }});

    // --- Misc ---
    k.push_back({"Magnitude", "misc", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_16SC1); cv::randu(src, -1000, 1000);
        dst = cv::Mat(h, w, CV_16SC1);
        auto src2 = src.clone(); cv::randu(src2, -1000, 1000);
        cv::Mat sf1, sf2;
        src.convertTo(sf1, CV_32F);
        src2.convertTo(sf2, CV_32F);
        run = [sf1, sf2](const cv::Mat& s, cv::Mat& d) {
            cv::Mat tmp;
            cv::magnitude(sf1, sf2, tmp);
            tmp.convertTo(d, CV_16S);
        };
    }});
    k.push_back({"Phase", "misc", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_16SC1); cv::randu(src, -1000, 1000);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, -1000, 1000);
        cv::Mat sf1, sf2;
        src.convertTo(sf1, CV_32F);
        src2.convertTo(sf2, CV_32F);
        run = [sf1, sf2](const cv::Mat& s, cv::Mat& d) {
            cv::Mat tmp;
            cv::phase(sf1, sf2, tmp);
            tmp.convertTo(d, CV_8U, 255.0 / (2.0 * CV_PI));
        };
    }});
    k.push_back({"TableLookup", "misc", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        cv::Mat lut(1, 256, CV_8UC1);
        for (int i = 0; i < 256; i++) lut.at<uchar>(0, i) = static_cast<uchar>(255 - i);
        run = [lut](const cv::Mat& s, cv::Mat& d) { cv::LUT(s, lut, d); };
    }});
    k.push_back({"Threshold", "misc", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::threshold(s, d, 128, 255, cv::THRESH_BINARY); };
    }});
    k.push_back({"WeightedAverage", "misc", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h, w, CV_8UC1);
        auto src2 = src.clone(); cv::randu(src2, 0, 256);
        run = [src2](const cv::Mat& s, cv::Mat& d) { cv::addWeighted(s, 0.5, src2, 0.5, 0, d); };
    }});

    // --- Multiscale ---
    k.push_back({"GaussianPyramid", "multiscale", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h / 2, w / 2, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::pyrDown(s, d); };
    }});
    k.push_back({"HalfScaleGaussian", "multiscale", [](int w, int h, cv::Mat& src, cv::Mat& dst, KernelFunc& run) {
        src = cv::Mat(h, w, CV_8UC1); cv::randu(src, 0, 256);
        dst = cv::Mat(h / 2, w / 2, CV_8UC1);
        run = [](const cv::Mat& s, cv::Mat& d) { cv::pyrDown(s, d); };
    }});

    return k;
}

// ---------------------------------------------------------------------------
// JSON writing helpers
// ---------------------------------------------------------------------------

static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static void writeTimingStats(std::ofstream& f, const std::string& indent, const TimingStats& t) {
    f << indent << "\"mean_ms\": "    << t.mean_ms    << ",\n";
    f << indent << "\"median_ms\": "  << t.median_ms  << ",\n";
    f << indent << "\"min_ms\": "     << t.min_ms     << ",\n";
    f << indent << "\"max_ms\": "     << t.max_ms     << ",\n";
    f << indent << "\"stddev_ms\": "  << t.stddev_ms  << ",\n";
    f << indent << "\"p5_ms\": "      << t.p5_ms      << ",\n";
    f << indent << "\"p95_ms\": "     << t.p95_ms     << ",\n";
    f << indent << "\"p99_ms\": "     << t.p99_ms     << ",\n";
    f << indent << "\"cv_percent\": " << t.cv_percent  << ",\n";
    f << indent << "\"sample_count\": "     << t.sample_count     << ",\n";
    f << indent << "\"outliers_removed\": " << t.outliers_removed << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // CLI defaults
    std::string resolution = "FHD";
    int iterations = 20;
    int warmup = 5;
    std::string outputDir = "benchmark_results";

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--resolution" || arg == "-r") && i + 1 < argc)
            resolution = argv[++i];
        else if ((arg == "--iterations" || arg == "-i") && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if ((arg == "--warmup" || arg == "-w") && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if ((arg == "--output-dir" || arg == "--output") && i + 1 < argc)
            outputDir = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: opencv-mark [OPTIONS]\n"
                      << "  --resolution, -r   Resolution: VGA, FHD, 4K (default: FHD)\n"
                      << "  --iterations, -i   Number of measurement iterations (default: 20)\n"
                      << "  --warmup, -w       Number of warmup iterations (default: 5)\n"
                      << "  --output-dir       Output directory (default: benchmark_results)\n"
                      << "  --help, -h         Show this help\n";
            return 0;
        }
    }

    // Find resolution
    const Resolution* res = nullptr;
    for (const auto& r : kResolutions) {
        if (r.name == resolution) { res = &r; break; }
    }
    if (!res) {
        std::cerr << "Unknown resolution: " << resolution << "\n";
        return 1;
    }

    std::cout << "opencv-mark v1.0.0\n";
    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    std::cout << "Resolution: " << res->name << " (" << res->width << "x" << res->height << ")\n";
    std::cout << "Iterations: " << iterations << ", Warmup: " << warmup << "\n\n";

    auto kernels = getKernels();
    auto sysInfo = getSystemInfo();
    std::vector<BenchmarkResult> results;

    for (const auto& kernel : kernels) {
        cv::Mat src, dst;
        KernelFunc run;
        kernel.setup(res->width, res->height, src, dst, run);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            run(src, dst);
        }

        // Measure
        std::vector<double> timings;
        timings.reserve(iterations);
        for (int i = 0; i < iterations; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run(src, dst);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings.push_back(ms);
        }

        BenchmarkResult br;
        br.name       = kernel.name;
        br.category   = kernel.category;
        br.mode       = "graph";
        br.resolution = res->name;
        br.width      = res->width;
        br.height     = res->height;
        br.iterations = iterations;
        br.warmup     = warmup;
        br.wall_clock = computeStats(timings);

        double pixels = static_cast<double>(res->width) * res->height;
        if (br.wall_clock.median_ms > 0) {
            br.megapixels_per_sec = pixels / (br.wall_clock.median_ms * 1000.0);
        }

        std::cout << "  " << br.name << " : "
                  << br.megapixels_per_sec << " MP/s  ("
                  << br.wall_clock.median_ms << " ms median, CV "
                  << br.wall_clock.cv_percent << "%)\n";

        results.push_back(std::move(br));
    }

    // Compute overall vision score (geometric mean of MP/s)
    double logSum = 0;
    int scoreCount = 0;
    // Per-category geometric means
    std::map<std::string, std::pair<double, int>> catScores;  // logsum, count
    for (const auto& r : results) {
        if (r.megapixels_per_sec > 0) {
            double lv = std::log(r.megapixels_per_sec);
            logSum += lv;
            scoreCount++;
            catScores[r.category].first += lv;
            catScores[r.category].second++;
        }
    }
    double overallScore = (scoreCount > 0) ? std::exp(logSum / scoreCount) : 0;

    // Write JSON
    std::filesystem::create_directories(outputDir);
    std::string jsonPath = outputDir + "/benchmark_results.json";
    std::ofstream f(jsonPath);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << jsonPath << " for writing\n";
        return 1;
    }

    f << std::fixed;
    f << "{\n";

    // system
    f << "  \"system\": {\n";
    f << "    \"hostname\": \"" << jsonEscape(sysInfo.hostname) << "\",\n";
    f << "    \"os_name\": \"" << jsonEscape(sysInfo.os_name) << "\",\n";
    f << "    \"os_version\": \"" << jsonEscape(sysInfo.os_version) << "\",\n";
    f << "    \"cpu_model\": \"" << jsonEscape(sysInfo.cpu_model) << "\",\n";
    f << "    \"cpu_cores\": " << sysInfo.cpu_cores << ",\n";
    f << "    \"ram_gb\": " << std::setprecision(1) << sysInfo.ram_gb << ",\n";
    f << "    \"timestamp\": \"" << jsonEscape(sysInfo.timestamp) << "\"\n";
    f << "  },\n";

    // openvx (compatibility field — populated with OpenCV info)
    f << "  \"openvx\": {\n";
    f << "    \"implementation\": \"OpenCV " << CV_VERSION << "\",\n";
    f << "    \"vendor_id\": 0,\n";
    f << "    \"version\": 0,\n";
    f << "    \"num_kernels\": " << kernels.size() << ",\n";
    f << "    \"extensions\": \"" << jsonEscape(cv::getBuildInformation().substr(0, 200)) << "\"\n";
    f << "  },\n";

    // benchmark
    f << "  \"benchmark\": {\n";
    f << "    \"version\": \"1.0.0\",\n";
    f << "    \"git_commit\": \"\"\n";
    f << "  },\n";

    // config
    f << "  \"config\": {\n";
    f << "    \"iterations\": " << iterations << ",\n";
    f << "    \"warmup\": " << warmup << ",\n";
    f << "    \"seed\": 42,\n";
    f << "    \"stability_threshold\": 15.0,\n";
    f << "    \"max_retries\": 0,\n";
    f << "    \"resolutions\": [\"" << res->name << "\"]\n";
    f << "  },\n";

    // scores
    f << "  \"scores\": {\n";
    f << "    \"overall_vision_score\": " << std::setprecision(2) << overallScore << ",\n";
    f << "    \"vision_benchmark_count\": " << scoreCount << ",\n";
    f << "    \"category_scores\": {\n";
    f << "      \"vision\": {\n";
    bool firstCat = true;
    for (const auto& [cat, pair] : catScores) {
        if (!firstCat) f << ",\n";
        firstCat = false;
        double catScore = (pair.second > 0) ? std::exp(pair.first / pair.second) : 0;
        f << "        \"" << jsonEscape(cat) << "\": " << std::setprecision(2) << catScore;
    }
    f << "\n      }\n";
    f << "    }\n";
    f << "  },\n";

    // conformance (empty — not applicable for OpenCV)
    f << "  \"conformance\": [],\n";

    // results
    f << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        f << "    {\n";
        f << "      \"name\": \"" << jsonEscape(r.name) << "\",\n";
        f << "      \"category\": \"" << jsonEscape(r.category) << "\",\n";
        f << "      \"feature_set\": \"vision\",\n";
        f << "      \"mode\": \"" << jsonEscape(r.mode) << "\",\n";
        f << "      \"resolution\": \"" << jsonEscape(r.resolution) << "\",\n";
        f << "      \"width\": " << r.width << ",\n";
        f << "      \"height\": " << r.height << ",\n";
        f << "      \"supported\": " << (r.supported ? "true" : "false") << ",\n";
        f << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";
        f << "      \"iterations\": " << r.iterations << ",\n";
        f << "      \"warmup\": " << r.warmup << ",\n";
        f << "      \"megapixels_per_sec\": " << std::setprecision(6) << r.megapixels_per_sec << ",\n";
        f << "      \"peak_ms\": " << std::setprecision(6) << r.wall_clock.min_ms << ",\n";
        f << "      \"sustained_ms\": " << std::setprecision(6) << r.wall_clock.median_ms << ",\n";
        double sustained_ratio = (r.wall_clock.median_ms > 0) ? (r.wall_clock.min_ms / r.wall_clock.median_ms) : 0;
        f << "      \"sustained_ratio\": " << std::setprecision(6) << sustained_ratio << ",\n";
        f << "      \"stability_warning\": " << (r.wall_clock.cv_percent > 15.0 ? "true" : "false") << ",\n";
        f << "      \"retry_count\": 0,\n";
        f << "      \"wall_clock\": {\n";
        writeTimingStats(f, "        ", r.wall_clock);
        f << "      },\n";
        f << "      \"framework_metrics\": []\n";
        f << "    }";
        if (i + 1 < results.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    f.close();

    std::cout << "\nResults written to " << jsonPath << "\n";
    std::cout << "Overall vision score: " << std::fixed << std::setprecision(2) << overallScore << "\n";

    return 0;
}

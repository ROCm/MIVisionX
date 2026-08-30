// openvx_remap_bench.cpp
// Benchmark OpenVX remap (CPU/GPU) against threaded OpenCV remap.
// Build CPU:
//   g++ -O3 -std=c++17 -fopenmp -I/home/kiriti/.openclaw/workspace/MIVisionX/amd_openvx/openvx/include \
//       openvx_remap_bench.cpp -L/home/kiriti/.openclaw/workspace/MIVisionX/build-local-cpu/lib -lopenvx \
//       -Wl,-rpath,/home/kiriti/.openclaw/workspace/MIVisionX/build-local-cpu/lib \
//       $(pkg-config --cflags --libs opencv4) -o openvx_remap_bench_cpu
// Build HIP:
//   g++ -O3 -std=c++17 -fopenmp -I/home/kiriti/.openclaw/workspace/MIVisionX/amd_openvx/openvx/include \
//       openvx_remap_bench.cpp -L/home/kiriti/.openclaw/workspace/MIVisionX/build-local-hip/lib -lopenvx \
//       -Wl,-rpath,/home/kiriti/.openclaw/workspace/MIVisionX/build-local-hip/lib \
//       $(pkg-config --cflags --libs opencv4) -o openvx_remap_bench_hip
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static double median(std::vector<double>& v) {
    size_t n = v.size();
    std::sort(v.begin(), v.end());
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
}

static void fill_image_u8(void* ptr, int w, int h, int stride, int channels, uint32_t color) {
    uint8_t* p = (uint8_t*)ptr;
    uint8_t c[4] = { (uint8_t)(color & 0xff), (uint8_t)((color >> 8) & 0xff),
                     (uint8_t)((color >> 16) & 0xff), (uint8_t)((color >> 24) & 0xff) };
    for (int y = 0; y < h; ++y) {
        uint8_t* row = p + (size_t)y * stride;
        for (int x = 0; x < w; ++x) {
            for (int k = 0; k < channels; ++k) row[x * channels + k] = c[k];
        }
    }
}

static void parallel_opencv_remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapx, const cv::Mat& mapy,
                                  int interpolation, int borderMode, const cv::Scalar& borderVal) {
    int nthreads = std::max(1, (int)std::thread::hardware_concurrency());
    int h = src.rows;
    int chunk = (h + nthreads - 1) / nthreads;
    std::vector<std::future<void>> futs;
    for (int t = 0; t < nthreads; ++t) {
        int y0 = t * chunk;
        int y1 = std::min(h, y0 + chunk);
        if (y0 >= y1) break;
        futs.push_back(std::async(std::launch::async, [&src, &dst, &mapx, &mapy, interpolation, borderMode, borderVal, y0, y1]() {
            cv::Mat srcRoi = src(cv::Range(y0, y1), cv::Range::all());
            cv::Mat mxRoi = mapx(cv::Range(y0, y1), cv::Range::all());
            cv::Mat myRoi = mapy(cv::Range(y0, y1), cv::Range::all());
            cv::Mat dstRoi = dst(cv::Range(y0, y1), cv::Range::all());
            cv::remap(srcRoi, dstRoi, mxRoi, myRoi, interpolation, borderMode, borderVal);
        }));
    }
    for (auto& f : futs) f.get();
}

static double bench_opencv(const cv::Mat& src, const cv::Mat& mapx, const cv::Mat& mapy,
                             int constant_border, uint32_t borderColor, int iterations) {
    cv::Mat dst(src.rows, src.cols, src.type());
    cv::Scalar borderVal(0, 0, 0, 0);
    int borderMode = constant_border ? cv::BORDER_CONSTANT : cv::BORDER_REPLICATE;
    for (int i = 0; i < std::max(1, iterations / 10); ++i) {
        parallel_opencv_remap(src, dst, mapx, mapy, cv::INTER_LINEAR, borderMode, borderVal);
    }
    std::vector<double> times;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            parallel_opencv_remap(src, dst, mapx, mapy, cv::INTER_LINEAR, borderMode, borderVal);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        times.push_back(ms / iterations);
    }
    return median(times);
}

static void setup_remap_table(vx_remap remap, int w, int h, bool constant_border) {
    vx_rectangle_t rect = { 0, 0, (vx_uint32)w, (vx_uint32)h };
    std::vector<vx_coordinates2df_t> coords((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y * w + x;
            if (constant_border) {
                coords[idx].x = coords[idx].y = -1.0f;
            } else {
                coords[idx].x = (vx_float32)(w - 1 - x);
                coords[idx].y = (vx_float32)(h - 1 - y);
            }
        }
    }
    vxCopyRemapPatch(remap, &rect, sizeof(vx_coordinates2df_t) * w, coords.data(),
                     VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

static double bench_openvx(vx_context ctx, int w, int h, vx_df_image fmt, int channels,
                           bool constant_border, uint8_t border_value, int iterations) {
    vx_image src = vxCreateImage(ctx, w, h, fmt);
    vx_image dst = vxCreateImage(ctx, w, h, fmt);
    vx_remap remap = vxCreateRemap(ctx, w, h, w, h);
    setup_remap_table(remap, w, h, constant_border);

    vx_rectangle_t rect = {0, 0, (vx_uint32)w, (vx_uint32)h};
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void* ptr = nullptr;
    vxMapImagePatch(src, &rect, 0, &map_id, &addr, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    uint32_t color = (channels == 1) ? 0xAA : (channels == 3 ? 0xAABBCC : 0xAABBCCCC);
    fill_image_u8(ptr, w, h, addr.stride_y, channels, color);
    vxUnmapImagePatch(src, map_id);

    vx_graph graph = vxCreateGraph(ctx);
    vx_node node = vxRemapNode(graph, src, remap, VX_INTERPOLATION_BILINEAR, dst);
    if (constant_border) {
        vx_border_t mode = { VX_BORDER_CONSTANT, border_value };
        vxSetNodeAttribute(node, VX_NODE_BORDER, &mode, sizeof(mode));
    }
    vxVerifyGraph(graph);

    for (int i = 0; i < std::max(1, iterations / 10); ++i) {
        vxProcessGraph(graph);
    }
    std::vector<double> times;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            vxProcessGraph(graph);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        times.push_back(ms / iterations);
    }
    double med = median(times);

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseRemap(&remap);
    vxReleaseImage(&dst);
    vxReleaseImage(&src);
    return med;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <backend:CPU|GPU> <w> <h> <u8|rgb|rgbx> <constant|mirror> [iters]\n", argv[0]);
        return 1;
    }
    const char* backend = argv[1];
    int W = atoi(argv[2]);
    int H = atoi(argv[3]);
    const char* fmtstr = argv[4];
    bool constant_border = strcmp(argv[5], "constant") == 0;
    int iterations = (argc > 6) ? atoi(argv[6]) : 200;

    int channels = 1;
    int cvType = CV_8UC1;
    vx_df_image fmt = VX_DF_IMAGE_U8;
    if (strcmp(fmtstr, "rgb") == 0) { channels = 3; cvType = CV_8UC3; fmt = VX_DF_IMAGE_RGB; }
    else if (strcmp(fmtstr, "rgbx") == 0) { channels = 4; cvType = CV_8UC4; fmt = VX_DF_IMAGE_RGBX; }

    if (strcmp(backend, "GPU") == 0) setenv("AGO_DEFAULT_TARGET", "GPU", 1);
    else setenv("AGO_DEFAULT_TARGET", "CPU", 1);

    vx_context ctx = vxCreateContext();
    if (!ctx) { fprintf(stderr, "vxCreateContext failed\n"); return 1; }

    double ovx_ms = bench_openvx(ctx, W, H, fmt, channels, constant_border, 0, iterations);
    vxReleaseContext(&ctx);

    cv::Mat src(H, W, cvType);
    uint32_t color = (channels == 1) ? 0xAA : (channels == 3 ? 0xAABBCC : 0xAABBCCCC);
    if (channels == 1) src.setTo(cv::Scalar(0xAA));
    else if (channels == 3) src.setTo(cv::Scalar(0xCC, 0xBB, 0xAA));
    else src.setTo(cv::Scalar(0xCC, 0xBB, 0xAA, 0xAA));

    cv::Mat mapx(H, W, CV_32FC1);
    cv::Mat mapy(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sx, sy;
            if (constant_border) sx = sy = -1.0f;
            else { sx = (float)(W - 1 - x); sy = (float)(H - 1 - y); }
            mapx.at<float>(y, x) = sx;
            mapy.at<float>(y, x) = sy;
        }
    }

    double oc_ms = bench_opencv(src, mapx, mapy, constant_border, 0, iterations);

    printf("%s,%dx%d,%s,%s,%.4f,%.4f,%.2f\n",
           backend, W, H, fmtstr, constant_border ? "constant" : "mirror", ovx_ms, oc_ms, oc_ms / ovx_ms);
    return 0;
}

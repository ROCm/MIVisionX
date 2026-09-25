/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//
// A surround-view lane pipeline for a multi-camera ADAS rig, built as the three
// stage chain the graph pipelining extension uses as its example and runnable
// several ways so the arrangements can be compared on identical work.
//
//   camera luma x N -> [ stage 0 ] -> surround -> [ stage 1 ] -> edges -> [ stage 2 ] -> lane mask
//                       undistort                  canny                  oriented filter bank
//                       denoise                                           morphological cleanup
//                       project to ground
//                       blend
//
// Stage 0 and stage 2 are many pixel-parallel operations per frame, which is
// what keeps a GPU busy for a useful length of time. Stage 1 is the detector,
// which on a desktop part runs faster on the CPU than on the GPU.
//
// Run one frame at a time and only one compute unit works while the other
// waits, so a frame costs the sum of the stages. Pipeline the stages and each
// one works on a different frame at the same time, so a frame costs only as
// much as the slowest stage. `--compare` measures the difference: it runs the
// same frames through every arrangement and prints one table.
//
// `--cameras` and `--filters` set how much work the two GPU stages carry, so
// the balance between the devices can be moved and its effect on the pipeline
// observed.
//

#include <VX/vx.h>
#include <VX/vx_khr_pipelining.h>
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#ifndef DEFAULT_WAITKEY_DELAY
#define DEFAULT_WAITKEY_DELAY 1
#endif

#define ERROR_CHECK_STATUS(call)                                                                \
    {                                                                                           \
        vx_status status_ = (call);                                                             \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

enum ExecutionMode
{
    MODE_CPU,     // one graph, every stage on the cpu, one frame at a time
    MODE_GPU,     // one graph, every stage on the gpu, one frame at a time
    MODE_SPLIT,   // one graph, stages split across the devices, still one frame at a time
    MODE_QUEUED,  // the split graph with queued parameters, capture overlaps the graph
    MODE_STAGED,  // one graph per stage, all three overlapped across frames
    MODE_BATCH,   // several frames handed over in one enqueue call
    MODE_STREAM   // the framework re-runs the graph on its own
};

enum StageIndex
{
    STAGE_SURROUND = 0,
    STAGE_DETECT = 1,
    STAGE_REFINE = 2,
    STAGE_COUNT = 3
};

// Every stage graph takes its inputs first and writes its output last, which is
// what makes the hand-off between them uniform. Stage 0 has one parameter per
// camera, the other two have a single input.
enum StageParam
{
    STAGE_INPUT = 0,
    STAGE_OUTPUT = 1
};

struct Config
{
    ExecutionMode mode = MODE_STAGED;
    bool compare = false;
    bool verify = false;
    std::string videoPath;
    int cameraId = -1;
    vx_uint32 width = 1280;
    vx_uint32 height = 720;
    vx_uint32 depth = 3;
    vx_uint32 cameras = 4;
    vx_uint32 filters = 4;
    vx_uint32 frames = 300;
    vx_uint32 batch = 4;
    bool preload = false;
    bool display = true;
    std::string dumpDir;
    vx_uint32 timeoutMs = 10000;
    vx_uint8 cannyLower = 20;
    vx_uint8 cannyUpper = 50;
    // Fractions of the raw frame trimmed before processing. A forward road
    // camera sees sky above and the car's hood or dashboard below; cropping
    // both keeps the pipeline on the road rather than on the vehicle interior.
    float cropTop = 0.08f;
    float cropBottom = 0.18f;
    bool manualSchedule = false;
    bool placementGiven = false;
    const char *placement[STAGE_COUNT] = {"gpu", "cpu", "gpu"};
    // When set, every lane mask a run produces is cloned here in output order,
    // so two modes can be compared frame for frame. --verify uses this.
    std::vector<cv::Mat> *laneSink = nullptr;
};

// One pool of buffers per hand-off point. The camera inputs need a pool per
// camera because each one is a graph parameter in its own right.
struct Pipeline
{
    vx_context context = nullptr;
    vx_graph graph = nullptr;
    vx_graph stage[STAGE_COUNT] = {nullptr, nullptr, nullptr};
    vx_remap undistort = nullptr;
    vx_matrix groundPlane = nullptr;
    vx_threshold hysteresis = nullptr;
    std::vector<vx_convolution> bank;
    vx_node trigger = nullptr;
    std::vector<vx_node> nodes;
    std::vector<std::vector<vx_image>> input; // [camera][slot]
    std::vector<vx_image> surround;
    std::vector<vx_image> edges;
    std::vector<vx_image> lanes;
};

struct Stats
{
    ExecutionMode mode = MODE_CPU;
    vx_uint32 frames = 0;
    double elapsedMs = 0.0;
    double graphMs = 0.0;
    double captureMs = 0.0;
    // Only a stage that is a graph of its own can be timed, so this is filled
    // in for the staged arrangement alone. Per-node timing would give the same
    // for the others, but this implementation answers a query for a node's
    // performance with its graph's, so there is nothing finer to be had.
    double stageMs[STAGE_COUNT] = {0.0, 0.0, 0.0};
    vx_uint32 graphCompleted = 0;
    vx_uint32 parameterConsumed = 0;
    vx_uint32 nodeErrors = 0;
};

static const vx_uint32 APP_VALUE_GRAPH_COMPLETED = 1;
static const vx_uint32 APP_VALUE_INPUT_CONSUMED = 2;

static const char *const STAGE_NAME[STAGE_COUNT] = {"surround", "detect", "refine"};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void VX_CALLBACK log_callback(vx_context, vx_reference, vx_status status, const vx_char string[])
{
    printf("LOG: [ status = %d ] %s\n", status, string);
    fflush(stdout);
}

static const char *mode_name(ExecutionMode mode)
{
    switch (mode)
    {
    case MODE_CPU:    return "cpu";
    case MODE_GPU:    return "gpu";
    case MODE_SPLIT:  return "split";
    case MODE_QUEUED: return "queued";
    case MODE_STAGED: return "staged";
    case MODE_BATCH:  return "batch";
    case MODE_STREAM: return "stream";
    }
    return "unknown";
}

static const char *mode_description(ExecutionMode mode)
{
    switch (mode)
    {
    case MODE_CPU:    return "one graph, every stage on the cpu, one frame at a time";
    case MODE_GPU:    return "one graph, every stage on the gpu, one frame at a time";
    case MODE_SPLIT:  return "one graph, stages split across the devices, one frame at a time";
    case MODE_QUEUED: return "the split graph with queued parameters, capture overlapped";
    case MODE_STAGED: return "one graph per stage, all three overlapped across frames";
    case MODE_BATCH:  return "several frames handed over in one enqueue call";
    case MODE_STREAM: return "the framework re-runs the graph on its own";
    }
    return "";
}

static const char *stage_target(const Config &cfg, StageIndex stage)
{
    if (cfg.mode == MODE_CPU)
        return "cpu";
    if (cfg.mode == MODE_GPU)
        return "gpu";
    return cfg.placement[stage];
}

static bool uses_queues(ExecutionMode mode)
{
    return mode == MODE_QUEUED || mode == MODE_BATCH;
}

static void print_usage(const char *argv0)
{
    printf("Usage:\n"
           "  %s --video <file> [options]\n"
           "  %s --live  <camera-id> [options]\n"
           "\n"
           "Options:\n"
           "  --compare                  run every arrangement on the same frames, then\n"
           "                             print one table (no display)\n"
           "  --verify                   check that each pipelined mode produces the same\n"
           "                             lane masks as the unpipelined reference (no display)\n"
           "  --mode <name>              cpu, gpu, split, queued, staged, batch, stream\n"
           "                               cpu     one graph, every stage on the cpu\n"
           "                               gpu     one graph, every stage on the gpu\n"
           "                               split   one graph, stages split across devices\n"
           "                               queued  the split graph, capture overlapped with it\n"
           "                               staged  one graph per stage, overlapped across frames\n"
           "                               batch   several frames per enqueue call\n"
           "                               stream  the framework re-runs the graph itself\n"
           "  --place <d,d,d>            device per stage for split, queued and staged,\n"
           "                             as surround,detect,refine (default gpu,cpu,gpu)\n"
           "  --cameras <N>              cameras in the rig, sets the stage 0 workload\n"
           "                             (default 4)\n"
           "  --filters <K>              matched filter orientations, sets the stage 2\n"
           "                             workload (default 4)\n"
           "  --schedule <auto|manual>   queueing mode for queued and staged (default auto)\n"
           "  --size <WxH>               processing resolution (default 1280x720)\n"
           "  --depth <N>                buffers per graph parameter (default 3)\n"
           "  --frames <N>               frames to process (default 300)\n"
           "  --batch <N>                frames per enqueue in batch mode (default 4)\n"
           "  --preload                  decode the frames into memory first, so the run\n"
           "                             measures the pipeline and not the video decoder\n"
           "  --no-display               skip the OpenCV window, for timing runs\n"
           "  --dump <dir>               write each lane mask as a png\n"
           "  --canny <lower>,<upper>    hysteresis thresholds (default 20,50)\n"
           "  --crop <top>,<bottom>      fractions trimmed off the frame before\n"
           "                             processing, to drop sky and the car interior\n"
           "                             (default 0.08,0.18)\n"
           "  --timeout <ms>             graph and event timeout (default 10000)\n"
           "\n"
           "Examples:\n"
           "  %s --video drive.mp4 --compare --size 1920x1080 --cameras 6 --filters 8\n"
           "  %s --video drive.mp4 --mode staged --depth 3\n"
           "  %s --video drive.mp4 --mode staged --place cpu,gpu,cpu\n",
           argv0, argv0, argv0, argv0, argv0);
}

static bool parse_mode(const std::string &name, ExecutionMode &mode)
{
    if (name == "cpu") mode = MODE_CPU;
    else if (name == "gpu") mode = MODE_GPU;
    else if (name == "split") mode = MODE_SPLIT;
    else if (name == "queued") mode = MODE_QUEUED;
    else if (name == "staged") mode = MODE_STAGED;
    else if (name == "batch") mode = MODE_BATCH;
    else if (name == "stream") mode = MODE_STREAM;
    else return false;
    return true;
}

static bool parse_placement(char *text, Config &cfg)
{
    const char *devices[STAGE_COUNT] = {nullptr, nullptr, nullptr};
    int found = 0;
    for (char *token = strtok(text, ","); token && found < STAGE_COUNT; token = strtok(nullptr, ","))
    {
        if (!strcmp(token, "cpu")) devices[found] = "cpu";
        else if (!strcmp(token, "gpu")) devices[found] = "gpu";
        else { printf("ERROR: --place expects cpu or gpu, got '%s'\n", token); return false; }
        found++;
    }
    if (found != STAGE_COUNT)
    {
        printf("ERROR: --place expects three devices, as surround,detect,refine\n");
        return false;
    }
    for (int i = 0; i < STAGE_COUNT; i++)
        cfg.placement[i] = devices[i];
    return true;
}

static bool parse_args(int argc, char *argv[], Config &cfg)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        bool hasValue = (i + 1) < argc;
        if (arg == "--video" && hasValue) cfg.videoPath = argv[++i];
        else if (arg == "--live" && hasValue) cfg.cameraId = atoi(argv[++i]);
        else if (arg == "--compare") cfg.compare = true;
        else if (arg == "--verify") cfg.verify = true;
        else if (arg == "--preload") cfg.preload = true;
        else if (arg == "--no-display") cfg.display = false;
        else if (arg == "--dump" && hasValue) cfg.dumpDir = argv[++i];
        else if (arg == "--cameras" && hasValue) cfg.cameras = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--filters" && hasValue) cfg.filters = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--depth" && hasValue) cfg.depth = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--frames" && hasValue) cfg.frames = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--batch" && hasValue) cfg.batch = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--timeout" && hasValue) cfg.timeoutMs = (vx_uint32)atoi(argv[++i]);
        else if (arg == "--place" && hasValue) { if (!parse_placement(argv[++i], cfg)) return false; cfg.placementGiven = true; }
        else if (arg == "--mode" && hasValue)
        {
            std::string name = argv[++i];
            if (!parse_mode(name, cfg.mode)) { printf("ERROR: unknown mode '%s'\n", name.c_str()); return false; }
        }
        else if (arg == "--schedule" && hasValue)
        {
            std::string name = argv[++i];
            if (name == "manual") cfg.manualSchedule = true;
            else if (name == "auto") cfg.manualSchedule = false;
            else { printf("ERROR: unknown schedule '%s'\n", name.c_str()); return false; }
        }
        else if (arg == "--size" && hasValue)
        {
            unsigned w = 0, h = 0;
            if (sscanf(argv[++i], "%ux%u", &w, &h) != 2 || !w || !h)
            {
                printf("ERROR: --size expects WxH\n");
                return false;
            }
            cfg.width = w;
            cfg.height = h;
        }
        else if (arg == "--canny" && hasValue)
        {
            unsigned lower = 0, upper = 0;
            if (sscanf(argv[++i], "%u,%u", &lower, &upper) != 2)
            {
                printf("ERROR: --canny expects <lower>,<upper>\n");
                return false;
            }
            cfg.cannyLower = (vx_uint8)lower;
            cfg.cannyUpper = (vx_uint8)upper;
        }
        else if (arg == "--crop" && hasValue)
        {
            float top = 0.0f, bottom = 0.0f;
            if (sscanf(argv[++i], "%f,%f", &top, &bottom) != 2 ||
                top < 0.0f || bottom < 0.0f || top + bottom >= 1.0f)
            {
                printf("ERROR: --crop expects <top>,<bottom> fractions with top+bottom < 1\n");
                return false;
            }
            cfg.cropTop = top;
            cfg.cropBottom = bottom;
        }
        else
        {
            printf("ERROR: unrecognized argument '%s'\n", arg.c_str());
            return false;
        }
    }
    if (cfg.videoPath.empty() && cfg.cameraId < 0)
    {
        printf("ERROR: one of --video or --live is required\n");
        return false;
    }
    if (cfg.depth < 1) cfg.depth = 1;
    if (cfg.cameras < 1) cfg.cameras = 1;
    if (cfg.filters < 1) cfg.filters = 1;
    if (cfg.batch < 1) cfg.batch = 1;
    if (cfg.frames == 0) cfg.frames = UINT32_MAX;
    if (cfg.compare || cfg.verify) cfg.display = false;
    if (cfg.placementGiven && !cfg.compare && !cfg.verify &&
        (cfg.mode == MODE_CPU || cfg.mode == MODE_GPU))
        printf("WARNING: --place is ignored in %s mode, which puts every stage on one device\n",
               mode_name(cfg.mode));
    return true;
}

// ---------------------------------------------------------------------------
// calibration data
// ---------------------------------------------------------------------------

//
// A barrel distortion model, the sort a calibrated wide-angle camera comes
// with. The table maps every undistorted pixel back to where the lens put it.
//
static vx_remap create_undistort_table(const Config &cfg, vx_context context)
{
    vx_remap remap = vxCreateRemap(context, cfg.width, cfg.height, cfg.width, cfg.height);
    ERROR_CHECK_OBJECT(remap);

    std::vector<vx_coordinates2df_t> table((size_t)cfg.width * cfg.height);
    const float cx = cfg.width * 0.5f;
    const float cy = cfg.height * 0.5f;
    const float norm = 1.0f / sqrtf(cx * cx + cy * cy);
    // A mild barrel model. A forward road camera is close to rectilinear, so
    // large coefficients only push the image edges inward and leave a black
    // ring the edge detector then fires on; these keep the correction gentle.
    const float k1 = -0.06f, k2 = 0.0f;

    for (vx_uint32 y = 0; y < cfg.height; y++)
    {
        for (vx_uint32 x = 0; x < cfg.width; x++)
        {
            const float dx = (x - cx) * norm;
            const float dy = (y - cy) * norm;
            const float r2 = dx * dx + dy * dy;
            const float scale = 1.0f + k1 * r2 + k2 * r2 * r2;
            vx_coordinates2df_t &point = table[(size_t)y * cfg.width + x];
            point.x = cx + dx * scale / norm;
            point.y = cy + dy * scale / norm;
        }
    }

    vx_rectangle_t rect = {0, 0, cfg.width, cfg.height};
    ERROR_CHECK_STATUS(vxCopyRemapPatch(remap, &rect,
                                        cfg.width * sizeof(vx_coordinates2df_t), table.data(),
                                        VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    return remap;
}

//
// The road ahead maps to a rectangle when viewed from above. warp_perspective
// reads the matrix as a destination-to-source mapping, so the homography to
// hand it is the one that sends ground-plane pixels back to the camera
// trapezoid.
//
static vx_matrix create_ground_plane(const Config &cfg, vx_context context)
{
    vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
    ERROR_CHECK_OBJECT(matrix);

    const float w = (float)cfg.width;
    const float h = (float)cfg.height;
    // Source trapezoid: a lane-width patch of road ahead, from just below the
    // horizon to the hood. Tuned against a forward road camera (comma10k).
    cv::Point2f camera[4] = {
        cv::Point2f(0.43f * w, 0.58f * h),
        cv::Point2f(0.57f * w, 0.58f * h),
        cv::Point2f(0.87f * w, 0.90f * h),
        cv::Point2f(0.13f * w, 0.90f * h)
    };
    // Destination spans the full frame, so the projection has no internal black
    // boundary for the edge detector to fire on; the warp border is replicated
    // rather than filled, keeping the strong lane ridges as the only edges.
    cv::Point2f ground[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(w, 0.0f),
        cv::Point2f(w, h),
        cv::Point2f(0.0f, h)
    };
    cv::Mat groundToCamera = cv::getPerspectiveTransform(ground, camera);

    // vx_matrix data is column major, so element (row, column) of the
    // homography lands at column * rows + row.
    vx_float32 data[9];
    for (int column = 0; column < 3; column++)
        for (int row = 0; row < 3; row++)
            data[column * 3 + row] = (vx_float32)groundToCamera.at<double>(row, column);

    ERROR_CHECK_STATUS(vxCopyMatrix(matrix, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    return matrix;
}

//
// A bank of oriented 9x9 kernels. Lane markings are bright ridges a few pixels
// wide, so each kernel answers "is there a ridge running this way here".
//
static std::vector<vx_convolution> create_filter_bank(const Config &cfg, vx_context context)
{
    std::vector<vx_convolution> bank;
    const double pi = 3.14159265358979323846;
    for (vx_uint32 f = 0; f < cfg.filters; f++)
    {
        const double angle = (pi * f) / cfg.filters;
        const double ca = cos(angle), sa = sin(angle);
        vx_int16 coefficients[81];
        for (int index = 0; index < 81; index++)
        {
            const double x = (index % 9) - 4.0;
            const double y = (index / 9) - 4.0;
            // distance from the ridge line running at this orientation
            const double across = x * sa - y * ca;
            const double along = fabs(x * ca + y * sa);
            const double ridge = exp(-across * across / 2.0) - 0.5 * exp(-across * across / 8.0);
            coefficients[index] = (vx_int16)lround(16.0 * ridge * (along <= 4.0 ? 1.0 : 0.0));
        }
        vx_convolution convolution = vxCreateConvolution(context, 9, 9);
        ERROR_CHECK_OBJECT(convolution);
        ERROR_CHECK_STATUS(vxCopyConvolutionCoefficients(convolution, coefficients,
                                                         VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        bank.push_back(convolution);
    }
    return bank;
}

// ---------------------------------------------------------------------------
// image transfer
// ---------------------------------------------------------------------------

static void write_luma(vx_image image, const cv::Mat &luma)
{
    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_HEIGHT, &height, sizeof(height)));
    vx_rectangle_t rect = {0, 0, width, height};
    vx_imagepatch_addressing_t layout;
    memset(&layout, 0, sizeof(layout));
    layout.stride_x = 1;
    layout.stride_y = (vx_int32)luma.step;
    ERROR_CHECK_STATUS(vxCopyImagePatch(image, &rect, 0, &layout, luma.data,
                                        VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
}

static void read_u8_image(vx_image image, cv::Mat &gray)
{
    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_HEIGHT, &height, sizeof(height)));
    if (gray.empty() || gray.cols != (int)width || gray.rows != (int)height || gray.type() != CV_8UC1)
        gray.create((int)height, (int)width, CV_8UC1);
    vx_rectangle_t rect = {0, 0, width, height};
    vx_imagepatch_addressing_t layout;
    memset(&layout, 0, sizeof(layout));
    layout.stride_x = 1;
    layout.stride_y = (vx_int32)gray.step;
    ERROR_CHECK_STATUS(vxCopyImagePatch(image, &rect, 0, &layout, gray.data,
                                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
}

static double graph_avg_ms(vx_graph graph)
{
    if (!graph)
        return 0.0;
    vx_perf_t perf;
    memset(&perf, 0, sizeof(perf));
    ERROR_CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_PERFORMANCE, &perf, sizeof(perf)));
    return perf.num ? (double)perf.avg * 1e-6 : 0.0;
}

// ---------------------------------------------------------------------------
// frame source
// ---------------------------------------------------------------------------

//
// The rig's cameras all deliver a frame per cycle. There is one video here, so
// every camera is given the same luma plane; the work per cycle is what the
// measurement is about.
//
class FrameSource
{
public:
    bool open(const Config &cfg)
    {
        m_size = cv::Size((int)cfg.width, (int)cfg.height);
        m_cropTop = cfg.cropTop;
        m_cropBottom = cfg.cropBottom;
        if (!cfg.videoPath.empty())
        {
            m_isFile = true;
            m_capture.open(cfg.videoPath);
        }
        else
        {
            m_isFile = false;
            m_capture.open(cfg.cameraId);
        }
        return m_capture.isOpened();
    }

    void rewind()
    {
        m_next = 0;
        if (m_isFile)
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    //
    // Decoding is host work of the same kind the pipeline is trying to overlap,
    // and at these resolutions it costs as much as the graph does, so a run
    // that decodes measures the decoder as much as the arrangement. Reading the
    // frames into memory first takes that out of the comparison. Enough of them
    // are kept to fill the budget below and then reused, which keeps the run
    // from turning into a measurement of the machine's memory instead.
    //
    bool preload(const Config &cfg)
    {
        static const size_t budgetBytes = 512u << 20;
        const size_t frameBytes = (size_t)cfg.width * cfg.height;
        size_t wanted = cfg.frames ? cfg.frames : 1;
        if (frameBytes && wanted > budgetBytes / frameBytes)
            wanted = budgetBytes / frameBytes;
        if (wanted < 1)
            wanted = 1;

        rewind();
        m_frames.clear();
        m_frames.reserve(wanted);
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < wanted; i++)
        {
            cv::Mat luma;
            if (!decode(luma))
                break;
            m_frames.push_back(luma);
        }
        m_preloadMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
        m_next = 0;
        return !m_frames.empty();
    }

    // Files are rewound when they run out so a requested frame count can always
    // be met, which keeps timing runs comparable across modes.
    bool next(cv::Mat &luma)
    {
        auto start = std::chrono::steady_clock::now();
        bool ok;
        if (!m_frames.empty())
        {
            // A shared header, not a copy: the caller only reads it.
            luma = m_frames[m_next % m_frames.size()];
            m_next++;
            ok = true;
        }
        else
        {
            ok = decode(luma);
        }
        m_captureMs += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start).count();
        m_captured++;
        return ok;
    }

    void resetTiming()
    {
        m_captureMs = 0.0;
        m_captured = 0;
    }

    double captureMsPerFrame() const { return m_captured ? m_captureMs / m_captured : 0.0; }
    double decodeMsPerFrame() const { return m_frames.empty() ? 0.0 : m_preloadMs / m_frames.size(); }
    size_t preloadedFrames() const { return m_frames.size(); }

private:
    bool decode(cv::Mat &luma)
    {
        cv::Mat frame;
        if (!m_capture.read(frame) || frame.empty())
        {
            if (!m_isFile)
                return false;
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!m_capture.read(frame) || frame.empty())
                return false;
        }
        // Trim the sky and the car interior before the resize, so the road
        // fills the processing frame the pipeline works on.
        int top = (int)(m_cropTop * frame.rows);
        int bottom = (int)(m_cropBottom * frame.rows);
        if (top + bottom < frame.rows)
            frame = frame(cv::Rect(0, top, frame.cols, frame.rows - top - bottom));
        if (frame.size() != m_size)
            cv::resize(frame, frame, m_size);
        cv::cvtColor(frame, luma, cv::COLOR_BGR2GRAY);
        return true;
    }

    cv::VideoCapture m_capture;
    bool m_isFile = false;
    cv::Size m_size;
    float m_cropTop = 0.0f;
    float m_cropBottom = 0.0f;
    std::vector<cv::Mat> m_frames;
    size_t m_next = 0;
    double m_preloadMs = 0.0;
    double m_captureMs = 0.0;
    vx_uint32 m_captured = 0;
};

// ---------------------------------------------------------------------------
// display
// ---------------------------------------------------------------------------

static bool abort_requested()
{
    int key = cv::waitKey(DEFAULT_WAITKEY_DELAY);
    if (key == ' ')
        key = cv::waitKey(0);
    return (key == 'q') || (key == 'Q') || (key == 27);
}

//
// The input frame is kept alongside the result only so the two can be shown or
// written together. A timing run keeps neither, which matters for the
// comparison: the unpipelined path holds one frame and copies nothing, so
// charging the arrangements that carry several frames for a copy their rivals
// never make would flatter the wrong one.
//
static cv::Mat retain_frame(const Config &cfg, const cv::Mat &luma)
{
    if (!cfg.display && cfg.dumpDir.empty())
        return cv::Mat();
    return luma.clone();
}

//
// The raw lane mask is what gets dumped, unannotated, so runs in different
// modes can be compared byte for byte.
//
static bool show_frame(const Config &cfg, const cv::Mat &luma, const cv::Mat &lanes,
                       const char *label)
{
    if (cfg.laneSink && !lanes.empty())
        cfg.laneSink->push_back(lanes.clone());

    if (!cfg.dumpDir.empty())
    {
        static vx_uint32 dumped = 0;
        char path[512];
        snprintf(path, sizeof(path), "%s/lanes_%05u.png", cfg.dumpDir.c_str(), dumped++);
        if (!cv::imwrite(path, lanes))
            printf("WARNING: could not write %s\n", path);
    }

    if (!cfg.display || luma.empty() || lanes.empty())
        return false;

    cv::Mat left, right, canvas;
    cv::cvtColor(luma, left, cv::COLOR_GRAY2BGR);
    cv::cvtColor(lanes, right, cv::COLOR_GRAY2BGR);
    cv::putText(left, label, cv::Point(8, 24), cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    cv::putText(left, "[ESC/Q] quit  [SPACE] pause", cv::Point(8, 48),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    cv::putText(right, "lane mask, ground plane", cv::Point(8, 24),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    cv::hconcat(left, right, canvas);
    if (canvas.cols > 1600)
        cv::resize(canvas, canvas, cv::Size(1600, canvas.rows * 1600 / canvas.cols));

    cv::imshow("adasPipeline", canvas);
    return abort_requested();
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

static void drain_events(vx_context context, Stats &stats)
{
    vx_event_t event;
    while (vxWaitEvent(context, &event, vx_true_e) == VX_SUCCESS)
    {
        switch (event.type)
        {
        case VX_EVENT_GRAPH_COMPLETED: stats.graphCompleted++; break;
        case VX_EVENT_GRAPH_PARAMETER_CONSUMED: stats.parameterConsumed++; break;
        case VX_EVENT_NODE_ERROR: stats.nodeErrors++; break;
        default: break;
        }
    }
}

// ---------------------------------------------------------------------------
// graph construction
// ---------------------------------------------------------------------------

static vx_uint32 buffer_count(const Config &cfg)
{
    switch (cfg.mode)
    {
    case MODE_CPU:
    case MODE_GPU:
    case MODE_SPLIT:
    case MODE_STREAM:
        return 1;
    case MODE_BATCH:
        return cfg.batch;
    default:
        return cfg.depth;
    }
}

static std::vector<vx_reference> as_references(const std::vector<vx_image> &images)
{
    std::vector<vx_reference> refs;
    refs.reserve(images.size());
    for (size_t i = 0; i < images.size(); i++)
        refs.push_back((vx_reference)images[i]);
    return refs;
}

static void add_graph_parameter(vx_graph graph, vx_node node, vx_uint32 index)
{
    vx_parameter parameter = vxGetParameterByIndex(node, index);
    ERROR_CHECK_OBJECT(parameter);
    ERROR_CHECK_STATUS(vxAddParameterToGraph(graph, parameter));
    ERROR_CHECK_STATUS(vxReleaseParameter(&parameter));
}

static void place(Pipeline &pipe, vx_node node, const char *target)
{
    ERROR_CHECK_OBJECT(node);
    ERROR_CHECK_STATUS(vxSetNodeTarget(node, VX_TARGET_STRING, target));
    pipe.nodes.push_back(node);
}

static void set_border_constant(vx_node node)
{
    vx_border_t border;
    memset(&border, 0, sizeof(border));
    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U8 = 0;
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));
}

//
// Every neighbourhood operator in the pipeline is given a constant border.
// Left at the default its border pixels are undefined, which the specification
// permits and this implementation makes use of by not writing them at all. The
// output then carries whatever the buffer it landed in happened to hold, and
// the buffers are recycled in a different order in every mode, so the same
// frame comes out slightly differently each run. Defining the border is what
// makes a frame come out identical whichever mode produced it, which is the
// only way to tell a pipelining fault from arithmetic that never matched.
//
static void place_bordered(Pipeline &pipe, vx_node node, const char *target)
{
    place(pipe, node, target);
    set_border_constant(node);
}

static void verify_or_die(vx_graph graph, const char *what)
{
    vx_status status = vxVerifyGraph(graph);
    if (status != VX_SUCCESS)
    {
        printf("ERROR: vxVerifyGraph(%s) failed with %d. On a build without GPU support use --mode cpu.\n",
               what, status);
        exit(1);
    }
}

static void create_shared_objects(const Config &cfg, Pipeline &pipe, vx_uint32 buffers, bool staged)
{
    pipe.context = vxCreateContext();
    ERROR_CHECK_OBJECT(pipe.context);
    vxRegisterLogCallback(pipe.context, log_callback, vx_false_e);

    pipe.input.resize(cfg.cameras);
    for (vx_uint32 slot = 0; slot < buffers; slot++)
    {
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        {
            vx_image in = vxCreateImage(pipe.context, cfg.width, cfg.height, VX_DF_IMAGE_U8);
            ERROR_CHECK_OBJECT(in);
            pipe.input[camera].push_back(in);
        }
        vx_image lane = vxCreateImage(pipe.context, cfg.width, cfg.height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(lane);
        pipe.lanes.push_back(lane);
        if (staged)
        {
            // The images between stages are real rather than virtual because
            // they are handed from one graph to the next.
            vx_image surround = vxCreateImage(pipe.context, cfg.width, cfg.height, VX_DF_IMAGE_U8);
            vx_image edge = vxCreateImage(pipe.context, cfg.width, cfg.height, VX_DF_IMAGE_U8);
            ERROR_CHECK_OBJECT(surround);
            ERROR_CHECK_OBJECT(edge);
            pipe.surround.push_back(surround);
            pipe.edges.push_back(edge);
        }
    }

    pipe.undistort = create_undistort_table(cfg, pipe.context);
    pipe.groundPlane = create_ground_plane(cfg, pipe.context);
    pipe.bank = create_filter_bank(cfg, pipe.context);

    pipe.hysteresis = vxCreateThresholdForImage(pipe.context, VX_THRESHOLD_TYPE_RANGE,
                                                VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(pipe.hysteresis);
    vx_pixel_value_t lower, upper;
    memset(&lower, 0, sizeof(lower));
    memset(&upper, 0, sizeof(upper));
    lower.U8 = cfg.cannyLower;
    upper.U8 = cfg.cannyUpper;
    ERROR_CHECK_STATUS(vxCopyThresholdRange(pipe.hysteresis, &lower, &upper,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxEnableEvents(pipe.context));
    ERROR_CHECK_STATUS(vxSetContextAttribute(pipe.context, VX_CONTEXT_EVENT_TIMEOUT,
                                             &cfg.timeoutMs, sizeof(cfg.timeoutMs)));
}

// The output of a stage, and the nodes a graph parameter has to be taken from.
struct StageNodes
{
    std::vector<vx_node> inputs; // one per camera for stage 0, otherwise one
    vx_node output = nullptr;    // node writing the stage output
    vx_uint32 outputIndex = 0;   // its parameter index for that output
};

//
// Stage 0. Each camera is undistorted, denoised and projected onto the ground
// plane, and the projections are blended into one surround view.
//
static StageNodes build_surround(const Config &cfg, Pipeline &pipe, vx_graph graph, vx_image output)
{
    const char *target = stage_target(cfg, STAGE_SURROUND);
    std::vector<vx_image> temporaries;
    StageNodes stage;

    // The projections are averaged, not summed. A surround view blends
    // overlapping ground projections; a saturating U8 sum of N of them would
    // blow the road out to white and bury the lane markings. Each projection is
    // accumulated into an S16 image (N*255 fits, so nothing saturates), then a
    // single convert back to U8 shifts right by ceil(log2 N) to form the mean.
    // With one camera there is nothing to average and the projection is written
    // straight to the output.
    vx_int32 blendShift = 0;
    while ((1u << blendShift) < cfg.cameras)
        blendShift++;

    vx_image accumulator = nullptr; // S16 running sum, once more than one camera
    vx_image firstProjection = nullptr;

    for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
    {
        vx_image undistorted = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
        vx_image denoised = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(undistorted);
        ERROR_CHECK_OBJECT(denoised);
        temporaries.push_back(undistorted);
        temporaries.push_back(denoised);

        vx_node remap = vxRemapNode(graph, pipe.input[camera][0], pipe.undistort,
                                    VX_INTERPOLATION_BILINEAR, undistorted);
        place_bordered(pipe, remap, target);
        stage.inputs.push_back(remap);
        place_bordered(pipe, vxGaussian3x3Node(graph, undistorted, denoised), target);

        // With one camera the warp writes straight to the stage output.
        const bool single = (cfg.cameras == 1);
        vx_image projected = single ? output
                                    : vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
        if (!single)
        {
            ERROR_CHECK_OBJECT(projected);
            temporaries.push_back(projected);
        }
        vx_node warp = vxWarpPerspectiveNode(graph, denoised, pipe.groundPlane,
                                             VX_INTERPOLATION_NEAREST_NEIGHBOR, projected);
        place_bordered(pipe, warp, target);
        stage.output = warp;
        stage.outputIndex = 3;

        if (single)
            continue;

        if (!firstProjection)
        {
            // Hold the first projection; the S16 sum starts when the second
            // arrives, since Add's S16 output takes two U8 inputs.
            firstProjection = projected;
            continue;
        }

        vx_image nextSum = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_S16);
        ERROR_CHECK_OBJECT(nextSum);
        temporaries.push_back(nextSum);
        if (!accumulator)
            place(pipe, vxAddNode(graph, firstProjection, projected, VX_CONVERT_POLICY_SATURATE, nextSum), target);
        else
            place(pipe, vxAddNode(graph, accumulator, projected, VX_CONVERT_POLICY_SATURATE, nextSum), target);
        accumulator = nextSum;
    }

    if (accumulator)
    {
        // Divide the running sum by the next power of two at or above N to make
        // the blended mean, writing the U8 result to the stage output.
        vx_scalar shift = vxCreateScalar(pipe.context, VX_TYPE_INT32, &blendShift);
        ERROR_CHECK_OBJECT(shift);
        vx_node mean = vxConvertDepthNode(graph, accumulator, output,
                                          VX_CONVERT_POLICY_SATURATE, shift);
        place(pipe, mean, target);
        ERROR_CHECK_STATUS(vxReleaseScalar(&shift));
        stage.output = mean;
        stage.outputIndex = 1;
    }

    for (size_t i = 0; i < temporaries.size(); i++)
        ERROR_CHECK_STATUS(vxReleaseImage(&temporaries[i]));
    return stage;
}

// Stage 1. The detector, one node, the expensive one.
static StageNodes build_detect(const Config &cfg, Pipeline &pipe, vx_graph graph,
                               vx_image input, vx_image output)
{
    StageNodes stage;
    vx_node canny = vxCannyEdgeDetectorNode(graph, input, pipe.hysteresis, 3, VX_NORM_L1, output);
    place(pipe, canny, stage_target(cfg, STAGE_DETECT));
    stage.inputs.push_back(canny);
    stage.output = canny;
    stage.outputIndex = 4;
    return stage;
}

//
// Stage 2. The edge mask is thickened, run through the oriented filter bank,
// the responses are combined, and the result is eroded back so the lane
// markings come out as solid strokes.
//
// The thickening is also what makes the stage safe to queue. A graph parameter
// stands for one parameter of one node, so when a queue swaps a buffer in only
// that one node is rebound. The filter bank has a node per orientation, so if
// they all read the queued image directly, every node but the first would keep
// reading whichever buffer was bound at build time. Feeding the bank from a
// single node that owns the queued input avoids that. Handing the same
// reference to one graph parameter per consumer would work too, at the cost of
// a queue each.
//
static StageNodes build_refine(const Config &cfg, Pipeline &pipe, vx_graph graph,
                               vx_image input, vx_image output)
{
    const char *target = stage_target(cfg, STAGE_REFINE);
    std::vector<vx_image> temporaries;
    StageNodes stage;
    vx_image combined = nullptr;

    vx_image thickened = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(thickened);
    temporaries.push_back(thickened);
    vx_node dilate = vxDilate3x3Node(graph, input, thickened);
    place_bordered(pipe, dilate, target);
    stage.inputs.push_back(dilate);

    for (size_t f = 0; f < pipe.bank.size(); f++)
    {
        vx_image response16 = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_S16);
        vx_image response8 = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(response16);
        ERROR_CHECK_OBJECT(response8);
        temporaries.push_back(response16);
        temporaries.push_back(response8);

        place_bordered(pipe, vxConvolveNode(graph, thickened, pipe.bank[f], response16), target);

        vx_int32 shift = 4;
        vx_scalar shiftScalar = vxCreateScalar(pipe.context, VX_TYPE_INT32, &shift);
        ERROR_CHECK_OBJECT(shiftScalar);
        place(pipe, vxConvertDepthNode(graph, response16, response8,
                                       VX_CONVERT_POLICY_SATURATE, shiftScalar), target);
        ERROR_CHECK_STATUS(vxReleaseScalar(&shiftScalar));

        if (!combined)
        {
            combined = response8;
        }
        else
        {
            vx_image merged = vxCreateVirtualImage(graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
            ERROR_CHECK_OBJECT(merged);
            temporaries.push_back(merged);
            place(pipe, vxAddNode(graph, combined, response8, VX_CONVERT_POLICY_SATURATE, merged), target);
            combined = merged;
        }
    }

    vx_node erode = vxErode3x3Node(graph, combined, output);
    place_bordered(pipe, erode, target);
    stage.output = erode;
    stage.outputIndex = 1;

    for (size_t i = 0; i < temporaries.size(); i++)
        ERROR_CHECK_STATUS(vxReleaseImage(&temporaries[i]));
    return stage;
}

//
// All three stages in one graph, with the images between them virtual.
//
static void build_single_graph(const Config &cfg, Pipeline &pipe)
{
    create_shared_objects(cfg, pipe, buffer_count(cfg), false);

    pipe.graph = vxCreateGraph(pipe.context);
    ERROR_CHECK_OBJECT(pipe.graph);

    vx_image surround = vxCreateVirtualImage(pipe.graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
    vx_image edges = vxCreateVirtualImage(pipe.graph, cfg.width, cfg.height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(surround);
    ERROR_CHECK_OBJECT(edges);

    StageNodes surroundStage = build_surround(cfg, pipe, pipe.graph, surround);
    build_detect(cfg, pipe, pipe.graph, surround, edges);
    StageNodes refineStage = build_refine(cfg, pipe, pipe.graph, edges, pipe.lanes[0]);
    pipe.trigger = refineStage.output;

    // one parameter per camera, then the lane mask
    for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        add_graph_parameter(pipe.graph, surroundStage.inputs[camera], 0);
    add_graph_parameter(pipe.graph, refineStage.output, refineStage.outputIndex);

    ERROR_CHECK_STATUS(vxReleaseImage(&surround));
    ERROR_CHECK_STATUS(vxReleaseImage(&edges));

    ERROR_CHECK_STATUS(vxRegisterEvent((vx_reference)pipe.graph, VX_EVENT_GRAPH_COMPLETED,
                                       0, APP_VALUE_GRAPH_COMPLETED));
    ERROR_CHECK_STATUS(vxRegisterEvent((vx_reference)pipe.graph, VX_EVENT_GRAPH_PARAMETER_CONSUMED,
                                       0, APP_VALUE_INPUT_CONSUMED));

    if (uses_queues(cfg.mode))
    {
        std::vector<std::vector<vx_reference>> refs;
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
            refs.push_back(as_references(pipe.input[camera]));
        refs.push_back(as_references(pipe.lanes));

        std::vector<vx_graph_parameter_queue_params_t> queues(refs.size());
        for (size_t i = 0; i < refs.size(); i++)
        {
            queues[i].graph_parameter_index = (vx_uint32)i;
            queues[i].refs_list_size = (vx_uint32)refs[i].size();
            queues[i].refs_list = refs[i].data();
        }
        const vx_enum scheduleMode = cfg.manualSchedule ? VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL
                                                        : VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO;
        ERROR_CHECK_STATUS(vxSetGraphScheduleConfig(pipe.graph, scheduleMode,
                                                    (vx_uint32)queues.size(), queues.data()));
    }

    if (cfg.mode == MODE_STREAM)
        ERROR_CHECK_STATUS(vxEnableGraphStreaming(pipe.graph, pipe.trigger));

    ERROR_CHECK_STATUS(vxSetGraphAttribute(pipe.graph, VX_GRAPH_TIMEOUT,
                                           &cfg.timeoutMs, sizeof(cfg.timeoutMs)));
    verify_or_die(pipe.graph, "single graph");
}

//
// The same three stages, one graph each. Every graph has its inputs queued
// first and its output queued last, and each gets its own executor, so while
// the detector runs on one frame the other two stages are working on others.
// The image a stage produces is consumed only by the next stage, which is what
// lets the queue swap a different buffer in for every execution without
// leaving a node bound to a stale one.
//
static void build_staged_graphs(const Config &cfg, Pipeline &pipe)
{
    create_shared_objects(cfg, pipe, cfg.depth, true);

    for (int i = 0; i < STAGE_COUNT; i++)
    {
        pipe.stage[i] = vxCreateGraph(pipe.context);
        ERROR_CHECK_OBJECT(pipe.stage[i]);
    }

    StageNodes surroundStage = build_surround(cfg, pipe, pipe.stage[STAGE_SURROUND], pipe.surround[0]);
    for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        add_graph_parameter(pipe.stage[STAGE_SURROUND], surroundStage.inputs[camera], 0);
    add_graph_parameter(pipe.stage[STAGE_SURROUND], surroundStage.output, surroundStage.outputIndex);

    StageNodes detectStage = build_detect(cfg, pipe, pipe.stage[STAGE_DETECT],
                                          pipe.surround[0], pipe.edges[0]);
    add_graph_parameter(pipe.stage[STAGE_DETECT], detectStage.inputs[0], 0);
    add_graph_parameter(pipe.stage[STAGE_DETECT], detectStage.output, detectStage.outputIndex);

    StageNodes refineStage = build_refine(cfg, pipe, pipe.stage[STAGE_REFINE],
                                          pipe.edges[0], pipe.lanes[0]);
    add_graph_parameter(pipe.stage[STAGE_REFINE], refineStage.inputs[0], 0);
    add_graph_parameter(pipe.stage[STAGE_REFINE], refineStage.output, refineStage.outputIndex);

    // Stage 0 has one queue per camera and one for its output; the other two
    // have one queue in and one out.
    const vx_enum scheduleMode = cfg.manualSchedule ? VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL
                                                    : VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO;
    std::vector<std::vector<vx_reference>> surroundRefs;
    for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        surroundRefs.push_back(as_references(pipe.input[camera]));
    surroundRefs.push_back(as_references(pipe.surround));

    std::vector<vx_graph_parameter_queue_params_t> surroundQueues(surroundRefs.size());
    for (size_t i = 0; i < surroundRefs.size(); i++)
    {
        surroundQueues[i].graph_parameter_index = (vx_uint32)i;
        surroundQueues[i].refs_list_size = (vx_uint32)surroundRefs[i].size();
        surroundQueues[i].refs_list = surroundRefs[i].data();
    }
    ERROR_CHECK_STATUS(vxSetGraphScheduleConfig(pipe.stage[STAGE_SURROUND], scheduleMode,
                                                (vx_uint32)surroundQueues.size(),
                                                surroundQueues.data()));

    std::vector<vx_reference> surroundList = as_references(pipe.surround);
    std::vector<vx_reference> edgeList = as_references(pipe.edges);
    std::vector<vx_reference> laneList = as_references(pipe.lanes);

    vx_graph_parameter_queue_params_t detectQueues[2];
    detectQueues[0] = {STAGE_INPUT, (vx_uint32)surroundList.size(), surroundList.data()};
    detectQueues[1] = {STAGE_OUTPUT, (vx_uint32)edgeList.size(), edgeList.data()};
    ERROR_CHECK_STATUS(vxSetGraphScheduleConfig(pipe.stage[STAGE_DETECT], scheduleMode,
                                                2, detectQueues));

    vx_graph_parameter_queue_params_t refineQueues[2];
    refineQueues[0] = {STAGE_INPUT, (vx_uint32)edgeList.size(), edgeList.data()};
    refineQueues[1] = {STAGE_OUTPUT, (vx_uint32)laneList.size(), laneList.data()};
    ERROR_CHECK_STATUS(vxSetGraphScheduleConfig(pipe.stage[STAGE_REFINE], scheduleMode,
                                                2, refineQueues));

    for (int i = 0; i < STAGE_COUNT; i++)
    {
        ERROR_CHECK_STATUS(vxRegisterEvent((vx_reference)pipe.stage[i], VX_EVENT_GRAPH_COMPLETED,
                                           0, APP_VALUE_GRAPH_COMPLETED));
        ERROR_CHECK_STATUS(vxSetGraphAttribute(pipe.stage[i], VX_GRAPH_TIMEOUT,
                                               &cfg.timeoutMs, sizeof(cfg.timeoutMs)));
        verify_or_die(pipe.stage[i], STAGE_NAME[i]);
    }
}

static void build_pipeline(const Config &cfg, Pipeline &pipe)
{
    if (cfg.mode == MODE_STAGED)
        build_staged_graphs(cfg, pipe);
    else
        build_single_graph(cfg, pipe);
}

static void release_images(std::vector<vx_image> &images)
{
    for (size_t i = 0; i < images.size(); i++)
        ERROR_CHECK_STATUS(vxReleaseImage(&images[i]));
    images.clear();
}

static void release_pipeline(Pipeline &pipe)
{
    for (size_t i = 0; i < pipe.nodes.size(); i++)
        ERROR_CHECK_STATUS(vxReleaseNode(&pipe.nodes[i]));
    pipe.nodes.clear();
    pipe.trigger = nullptr;
    if (pipe.graph)
        ERROR_CHECK_STATUS(vxReleaseGraph(&pipe.graph));
    for (int i = 0; i < STAGE_COUNT; i++)
        if (pipe.stage[i])
            ERROR_CHECK_STATUS(vxReleaseGraph(&pipe.stage[i]));
    for (size_t camera = 0; camera < pipe.input.size(); camera++)
        release_images(pipe.input[camera]);
    pipe.input.clear();
    release_images(pipe.surround);
    release_images(pipe.edges);
    release_images(pipe.lanes);
    for (size_t i = 0; i < pipe.bank.size(); i++)
        ERROR_CHECK_STATUS(vxReleaseConvolution(&pipe.bank[i]));
    pipe.bank.clear();
    if (pipe.hysteresis)
        ERROR_CHECK_STATUS(vxReleaseThreshold(&pipe.hysteresis));
    if (pipe.groundPlane)
        ERROR_CHECK_STATUS(vxReleaseMatrix(&pipe.groundPlane));
    if (pipe.undistort)
        ERROR_CHECK_STATUS(vxReleaseRemap(&pipe.undistort));
    if (pipe.context)
        ERROR_CHECK_STATUS(vxReleaseContext(&pipe.context));
    pipe = Pipeline();
}

// ---------------------------------------------------------------------------
// queue helpers
// ---------------------------------------------------------------------------

static void enqueue(vx_graph graph, vx_uint32 parameterIndex, vx_image image)
{
    vx_reference ref = (vx_reference)image;
    ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, parameterIndex, &ref, 1));
}

static vx_image dequeue(vx_graph graph, vx_uint32 parameterIndex)
{
    vx_reference ref = nullptr;
    vx_uint32 count = 0;
    ERROR_CHECK_STATUS(vxGraphParameterDequeueDoneRef(graph, parameterIndex, &ref, 1, &count));
    if (count != 1 || !ref)
    {
        printf("ERROR: graph parameter %u returned %u references\n", parameterIndex, count);
        exit(1);
    }
    return (vx_image)ref;
}

static vx_uint32 index_of(const std::vector<vx_image> &buffers, vx_image image)
{
    for (size_t i = 0; i < buffers.size(); i++)
        if (buffers[i] == image)
            return (vx_uint32)i;
    printf("ERROR: a dequeued reference does not belong to the buffer pool\n");
    exit(1);
}

static void schedule_if_manual(const Config &cfg, vx_graph graph)
{
    if (cfg.manualSchedule)
        ERROR_CHECK_STATUS(vxScheduleGraph(graph));
}

// ---------------------------------------------------------------------------
// execution modes
// ---------------------------------------------------------------------------

static void run_unpipelined(const Config &cfg, Pipeline &pipe, FrameSource &source, Stats &stats)
{
    cv::Mat luma, lanes;
    char label[160];
    auto start = std::chrono::steady_clock::now();

    for (vx_uint32 i = 0; i < cfg.frames; i++)
    {
        if (!source.next(luma))
            break;
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
            write_luma(pipe.input[camera][0], luma);
        ERROR_CHECK_STATUS(vxProcessGraph(pipe.graph));
        stats.frames++;

        read_u8_image(pipe.lanes[0], lanes);
        drain_events(pipe.context, stats);
        snprintf(label, sizeof(label), "%s  frame %u", mode_name(cfg.mode), stats.frames);
        if (show_frame(cfg, luma, lanes, label))
            break;
    }

    stats.elapsedMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
}

//
// The host reads and converts the next frame while the graph is still working
// on the frames already handed over.
//
static void run_queued(const Config &cfg, Pipeline &pipe, FrameSource &source, Stats &stats)
{
    const vx_uint32 laneParam = cfg.cameras;
    std::vector<cv::Mat> hostFrames(cfg.depth);
    cv::Mat luma, lanes;
    char label[160];
    vx_uint32 submitted = 0;

    auto start = std::chrono::steady_clock::now();

    for (vx_uint32 slot = 0; slot < cfg.depth && submitted < cfg.frames; slot++)
    {
        if (!source.next(luma))
            break;
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        {
            write_luma(pipe.input[camera][slot], luma);
            enqueue(pipe.graph, camera, pipe.input[camera][slot]);
        }
        hostFrames[slot] = retain_frame(cfg, luma);
        enqueue(pipe.graph, laneParam, pipe.lanes[slot]);
        submitted++;
        schedule_if_manual(cfg, pipe.graph);
    }

    bool stop = false;
    while (stats.frames < submitted)
    {
        vx_image doneLanes = dequeue(pipe.graph, laneParam);
        std::vector<vx_image> doneInputs(cfg.cameras);
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
            doneInputs[camera] = dequeue(pipe.graph, camera);
        const vx_uint32 slot = index_of(pipe.input[0], doneInputs[0]);
        stats.frames++;

        read_u8_image(doneLanes, lanes);
        drain_events(pipe.context, stats);
        cv::Mat processed = hostFrames[slot];

        if (!stop && submitted < cfg.frames && source.next(luma))
        {
            for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
            {
                write_luma(doneInputs[camera], luma);
                enqueue(pipe.graph, camera, doneInputs[camera]);
            }
            hostFrames[slot] = retain_frame(cfg, luma);
            enqueue(pipe.graph, laneParam, doneLanes);
            submitted++;
            schedule_if_manual(cfg, pipe.graph);
        }

        snprintf(label, sizeof(label), "queued  frame %u  depth %u  in flight %u",
                 stats.frames, cfg.depth, submitted - stats.frames);
        if (!stop && show_frame(cfg, processed, lanes, label))
            stop = true;
    }

    stats.elapsedMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
}

//
// A stage becomes an execution only once it holds both something to read and
// somewhere to write, and those two arrive at different times: a stage is given
// its input as soon as the stage before it finishes, but gets its output buffer
// back only when the stage after it releases one. Counting the two separately
// is what tells the driver when an execution is actually due, which QUEUE_MANUAL
// needs because vxScheduleGraph rejects a graph whose queues are not all ready.
//
struct StageReady
{
    vx_uint32 input = 0;
    vx_uint32 output = 0;
};

//
// One graph per stage, driven like a shift register: each pass collects what a
// stage finished during the previous pass and only then hands it the next
// frame. Nothing is waited on in the pass that submitted it, so all three
// graphs and the host are working on different frames at once.
//
static void run_staged(const Config &cfg, Pipeline &pipe, FrameSource &source, Stats &stats)
{
    vx_graph surroundStage = pipe.stage[STAGE_SURROUND];
    vx_graph detectStage = pipe.stage[STAGE_DETECT];
    vx_graph refineStage = pipe.stage[STAGE_REFINE];
    const vx_uint32 surroundOutParam = cfg.cameras;

    std::deque<cv::Mat> inFlightFrames;
    StageReady ready[STAGE_COUNT];
    vx_uint32 inFlight[STAGE_COUNT] = {0, 0, 0};
    cv::Mat luma, lanes;
    char label[160];
    vx_uint32 submitted = 0;
    bool stop = false;

    // Turns whatever has become complete into executions, so that a stage is
    // only ever scheduled once every one of its queues holds a reference, which
    // is what QUEUE_MANUAL requires.
    //
    // One request is made per pass rather than per execution, because the mode
    // runs every complete set it can find and is serviced on the graph's own
    // thread rather than in this one.
    auto dispatch = [&]() {
        for (int i = 0; i < STAGE_COUNT; i++)
        {
            vx_uint32 due = 0;
            while (ready[i].input && ready[i].output)
            {
                ready[i].input--;
                ready[i].output--;
                inFlight[i]++;
                due++;
            }
            if (due)
                schedule_if_manual(cfg, pipe.stage[i]);
        }
    };

    auto start = std::chrono::steady_clock::now();

    for (vx_uint32 slot = 0; slot < cfg.depth && submitted < cfg.frames; slot++)
    {
        if (!source.next(luma))
            break;
        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        {
            write_luma(pipe.input[camera][slot], luma);
            enqueue(surroundStage, camera, pipe.input[camera][slot]);
        }
        inFlightFrames.push_back(retain_frame(cfg, luma));
        enqueue(surroundStage, surroundOutParam, pipe.surround[slot]);
        enqueue(detectStage, STAGE_OUTPUT, pipe.edges[slot]);
        enqueue(refineStage, STAGE_OUTPUT, pipe.lanes[slot]);
        submitted++;
        ready[STAGE_SURROUND].input++;
        ready[STAGE_SURROUND].output++;
        ready[STAGE_DETECT].output++;
        ready[STAGE_REFINE].output++;
        dispatch();
    }

    while (stats.frames < submitted)
    {
        if (inFlight[STAGE_REFINE])
        {
            vx_image doneLanes = dequeue(refineStage, STAGE_OUTPUT);
            vx_image freeEdges = dequeue(refineStage, STAGE_INPUT);
            inFlight[STAGE_REFINE]--;
            enqueue(detectStage, STAGE_OUTPUT, freeEdges);
            ready[STAGE_DETECT].output++;
            stats.frames++;

            read_u8_image(doneLanes, lanes);
            cv::Mat processed = inFlightFrames.front();
            inFlightFrames.pop_front();
            snprintf(label, sizeof(label), "staged  frame %u  depth %u  in flight %u",
                     stats.frames, cfg.depth, submitted - stats.frames);
            if (!stop && show_frame(cfg, processed, lanes, label))
                stop = true;
            enqueue(refineStage, STAGE_OUTPUT, doneLanes);
            ready[STAGE_REFINE].output++;
        }

        if (inFlight[STAGE_DETECT])
        {
            vx_image doneEdges = dequeue(detectStage, STAGE_OUTPUT);
            vx_image freeSurround = dequeue(detectStage, STAGE_INPUT);
            inFlight[STAGE_DETECT]--;
            enqueue(surroundStage, surroundOutParam, freeSurround);
            ready[STAGE_SURROUND].output++;
            enqueue(refineStage, STAGE_INPUT, doneEdges);
            ready[STAGE_REFINE].input++;
        }

        if (inFlight[STAGE_SURROUND])
        {
            vx_image doneSurround = dequeue(surroundStage, surroundOutParam);
            std::vector<vx_image> freeInputs(cfg.cameras);
            for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
                freeInputs[camera] = dequeue(surroundStage, camera);
            inFlight[STAGE_SURROUND]--;
            enqueue(detectStage, STAGE_INPUT, doneSurround);
            ready[STAGE_DETECT].input++;

            if (!stop && submitted < cfg.frames && source.next(luma))
            {
                for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
                {
                    write_luma(freeInputs[camera], luma);
                    enqueue(surroundStage, camera, freeInputs[camera]);
                }
                inFlightFrames.push_back(retain_frame(cfg, luma));
                ready[STAGE_SURROUND].input++;
                submitted++;
            }
        }

        dispatch();
        drain_events(pipe.context, stats);
    }

    stats.elapsedMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
    for (int i = 0; i < STAGE_COUNT; i++)
        stats.stageMs[i] = graph_avg_ms(pipe.stage[i]);
}

//
// Batch mode hands several frames over in a single enqueue call and collects
// the whole set afterwards.
//
static void run_batch(const Config &cfg, Pipeline &pipe, FrameSource &source, Stats &stats)
{
    const vx_uint32 laneParam = cfg.cameras;
    const vx_uint32 batch = cfg.batch;
    std::vector<cv::Mat> hostFrames(batch);
    std::vector<vx_reference> laneRefs(batch);
    cv::Mat lanes;
    char label[160];

    for (vx_uint32 i = 0; i < batch; i++)
        laneRefs[i] = (vx_reference)pipe.lanes[i];

    auto start = std::chrono::steady_clock::now();
    bool stop = false;

    while (!stop && stats.frames < cfg.frames)
    {
        vx_uint32 filled = 0;
        for (vx_uint32 slot = 0; slot < batch && stats.frames + filled < cfg.frames; slot++)
        {
            if (!source.next(hostFrames[slot]))
                break;
            for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
                write_luma(pipe.input[camera][slot], hostFrames[slot]);
            filled++;
        }
        if (filled == 0)
            break;

        for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        {
            std::vector<vx_reference> refs(filled);
            for (vx_uint32 slot = 0; slot < filled; slot++)
                refs[slot] = (vx_reference)pipe.input[camera][slot];
            ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(pipe.graph, camera,
                                                               refs.data(), filled));
        }
        ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(pipe.graph, laneParam,
                                                           laneRefs.data(), filled));
        // One call covers the whole batch: QUEUE_MANUAL executes every complete
        // set of ready references it finds.
        schedule_if_manual(cfg, pipe.graph);

        for (vx_uint32 collected = 0; collected < filled; collected++)
        {
            vx_image doneLanes = dequeue(pipe.graph, laneParam);
            vx_uint32 slot = 0;
            for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
            {
                vx_image doneInput = dequeue(pipe.graph, camera);
                if (camera == 0)
                    slot = index_of(pipe.input[0], doneInput);
            }
            stats.frames++;
            read_u8_image(doneLanes, lanes);
            snprintf(label, sizeof(label), "batch  %u of %u  frame %u",
                     collected + 1, filled, stats.frames);
            if (!stop && show_frame(cfg, hostFrames[slot], lanes, label))
                stop = true;
        }
        drain_events(pipe.context, stats);
    }

    stats.elapsedMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
}

//
// Streaming hands the graph to the framework, which re-executes it until asked
// to stop. Without queued parameters the host and the streaming thread share
// one set of images, so writing the next frame is paced by the completion
// event to keep that window as small as the mode allows.
//
static void run_stream(const Config &cfg, Pipeline &pipe, FrameSource &source, Stats &stats)
{
    cv::Mat luma, lanes;
    char label[160];

    if (!source.next(luma))
    {
        printf("ERROR: no frames available\n");
        return;
    }
    for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
        write_luma(pipe.input[camera][0], luma);

    auto start = std::chrono::steady_clock::now();
    ERROR_CHECK_STATUS(vxStartGraphStreaming(pipe.graph));

    while (stats.frames < cfg.frames)
    {
        vx_event_t event;
        vx_status status = vxWaitEvent(pipe.context, &event, vx_false_e);
        if (status != VX_SUCCESS)
        {
            printf("ERROR: waiting for a graph event failed with %d\n", status);
            break;
        }
        if (event.type == VX_EVENT_NODE_ERROR)
        {
            stats.nodeErrors++;
            break;
        }
        if (event.type != VX_EVENT_GRAPH_COMPLETED)
            continue;

        stats.graphCompleted++;
        stats.frames++;
        read_u8_image(pipe.lanes[0], lanes);
        snprintf(label, sizeof(label), "stream  completions %u", stats.graphCompleted);
        bool stop = show_frame(cfg, luma, lanes, label);

        if (source.next(luma))
            for (vx_uint32 camera = 0; camera < cfg.cameras; camera++)
                write_luma(pipe.input[camera][0], luma);
        if (stop)
            break;
    }

    ERROR_CHECK_STATUS(vxStopGraphStreaming(pipe.graph));
    stats.elapsedMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
    drain_events(pipe.context, stats);
}

// ---------------------------------------------------------------------------
// driving one configuration
// ---------------------------------------------------------------------------

static Stats run_one(const Config &cfg, FrameSource &source)
{
    Pipeline pipe;
    build_pipeline(cfg, pipe);

    Stats stats;
    stats.mode = cfg.mode;
    source.resetTiming();
    switch (cfg.mode)
    {
    case MODE_CPU:
    case MODE_GPU:
    case MODE_SPLIT:  run_unpipelined(cfg, pipe, source, stats); break;
    case MODE_QUEUED: run_queued(cfg, pipe, source, stats); break;
    case MODE_STAGED: run_staged(cfg, pipe, source, stats); break;
    case MODE_BATCH:  run_batch(cfg, pipe, source, stats); break;
    case MODE_STREAM: run_stream(cfg, pipe, source, stats); break;
    }
    stats.graphMs = graph_avg_ms(pipe.graph);
    stats.captureMs = source.captureMsPerFrame();

    release_pipeline(pipe);
    return stats;
}

static void report(const Config &cfg, const Stats &stats)
{
    const double seconds = stats.elapsedMs / 1000.0;
    printf("\n");
    printf("mode                  : %s (%s)\n", mode_name(stats.mode), mode_description(stats.mode));
    printf("stage placement       : surround %s, detect %s, refine %s\n",
           stage_target(cfg, STAGE_SURROUND), stage_target(cfg, STAGE_DETECT),
           stage_target(cfg, STAGE_REFINE));
    printf("workload              : %u cameras, %u filter orientations, %ux%u\n",
           cfg.cameras, cfg.filters, cfg.width, cfg.height);
    if (stats.mode == MODE_QUEUED || stats.mode == MODE_STAGED)
        printf("queue depth           : %u  (%s scheduling)\n", cfg.depth,
               cfg.manualSchedule ? "manual" : "auto");
    if (stats.mode == MODE_BATCH)
        printf("frames per enqueue    : %u\n", cfg.batch);
    printf("frames                : %u\n", stats.frames);
    printf("wall time             : %.1f ms\n", stats.elapsedMs);
    if (seconds > 0.0)
        printf("throughput            : %.1f fps\n", stats.frames / seconds);
    printf("capture               : %.3f ms per frame%s\n", stats.captureMs,
           cfg.preload ? " (preloaded)" : "");
    if (stats.graphMs > 0.0)
        printf("graph time            : %.3f ms per frame\n", stats.graphMs);
    if (stats.mode == MODE_STAGED)
        for (int i = 0; i < STAGE_COUNT; i++)
            printf("stage %-9s (%s) : %.3f ms per frame\n", STAGE_NAME[i],
                   stage_target(cfg, (StageIndex)i), stats.stageMs[i]);
    printf("events completed      : %u\n", stats.graphCompleted);
    printf("events input consumed : %u\n", stats.parameterConsumed);
    if (stats.nodeErrors)
        printf("events node error     : %u\n", stats.nodeErrors);
}

static void run_comparison(Config cfg, FrameSource &source)
{
    const ExecutionMode ladder[] = {MODE_CPU, MODE_GPU, MODE_SPLIT, MODE_QUEUED,
                                    MODE_STAGED, MODE_BATCH, MODE_STREAM};
    const size_t count = sizeof(ladder) / sizeof(ladder[0]);
    std::vector<Stats> results;

    for (size_t i = 0; i < count; i++)
    {
        Config runCfg = cfg;
        runCfg.mode = ladder[i];
        source.rewind();
        printf("running %-7s %s\n", mode_name(ladder[i]), mode_description(ladder[i]));
        fflush(stdout);
        results.push_back(run_one(runCfg, source));
    }

    Config stagedCfg = cfg;
    stagedCfg.mode = MODE_STAGED;
    printf("\n%ux%u, %u cameras, %u filter orientations, %u frames per mode, queue depth %u\n",
           cfg.width, cfg.height, cfg.cameras, cfg.filters, cfg.frames, cfg.depth);
    printf("placement for split, queued and staged: surround %s, detect %s, refine %s\n\n",
           stage_target(stagedCfg, STAGE_SURROUND), stage_target(stagedCfg, STAGE_DETECT),
           stage_target(stagedCfg, STAGE_REFINE));
    printf("  %-7s %-8s %-9s %-9s %-8s  %s\n", "mode", "fps", "ms/frame", "capture",
           "vs cpu", "what runs at the same time");

    double baseline = 0.0;
    for (size_t i = 0; i < results.size(); i++)
    {
        const Stats &s = results[i];
        const double fps = s.frames / (s.elapsedMs / 1000.0);
        if (i == 0)
            baseline = fps;
        const char *overlap = "nothing, one stage at a time";
        switch (s.mode)
        {
        case MODE_SPLIT:  overlap = "nothing, the devices take turns"; break;
        case MODE_QUEUED: overlap = "capture against the graph"; break;
        case MODE_STAGED: overlap = "capture and all three stages, on different frames"; break;
        case MODE_BATCH:  overlap = "the frames in one enqueue, one graph"; break;
        case MODE_STREAM: overlap = "the framework re-runs the graph itself"; break;
        default: break;
        }
        printf("  %-7s %-8.1f %-9.2f %-9.2f %-8.2fx %s\n", mode_name(s.mode), fps,
               s.elapsedMs / s.frames, s.captureMs, fps / baseline, overlap);
    }

    if (cfg.preload)
        printf("\n  frames were decoded up front, so the capture column is a memory read\n");

    size_t stagedIndex = 0;
    for (size_t i = 0; i < results.size(); i++)
        if (results[i].mode == MODE_STAGED)
            stagedIndex = i;
    const Stats &staged = results[stagedIndex];
    double total = 0.0;
    printf("\n  staged stage graphs: ");
    for (int i = 0; i < STAGE_COUNT; i++)
    {
        printf("%s (%s) %.2f ms%s", STAGE_NAME[i], stage_target(stagedCfg, (StageIndex)i),
               staged.stageMs[i], (i + 1 < STAGE_COUNT) ? ", " : "\n");
        total += staged.stageMs[i];
    }

    // How much of the work is in the air at once, which is the whole point of
    // the arrangement. It is deliberately not phrased as a ceiling: the two
    // stages sharing the gpu still overlap on it, so a sum per device predicts
    // a floor the pipeline goes straight through.
    const double achieved = staged.elapsedMs / staged.frames;
    if (achieved > 0.0)
        printf("  they sum to %.2f ms of graph time per frame and the pipeline delivered a\n"
               "  frame every %.2f ms, so %.2fx of that work was in flight at once\n",
               total, achieved, total / achieved);
    printf("\n");
}

//
// Runs an unpipelined reference and each pipelined mode over the same frames,
// then compares the lane masks frame for frame. The outer six pixels are
// excluded because the neighbourhood operators leave their border pixels
// undefined by specification, so they differ run to run even in one mode; a
// pipelining fault shows up as a difference in the interior, which this counts.
//
static void run_verify(Config cfg, FrameSource &source)
{
    const vx_uint32 border = 6;
    if (cfg.frames > 50)
        cfg.frames = 50;

    std::vector<cv::Mat> reference;
    Config refCfg = cfg;
    refCfg.mode = MODE_SPLIT;
    refCfg.laneSink = &reference;
    source.rewind();
    printf("verify: reference is split over %u frames, placement surround %s, detect %s, refine %s\n",
           cfg.frames, stage_target(refCfg, STAGE_SURROUND), stage_target(refCfg, STAGE_DETECT),
           stage_target(refCfg, STAGE_REFINE));
    fflush(stdout);
    run_one(refCfg, source);

    const ExecutionMode candidates[] = {MODE_QUEUED, MODE_STAGED, MODE_BATCH};
    bool allMatched = true;
    for (size_t c = 0; c < sizeof(candidates) / sizeof(candidates[0]); c++)
    {
        std::vector<cv::Mat> masks;
        Config runCfg = cfg;
        runCfg.mode = candidates[c];
        runCfg.laneSink = &masks;
        source.rewind();
        run_one(runCfg, source);

        const size_t pairs = std::min(reference.size(), masks.size());
        vx_uint32 matched = 0;
        for (size_t i = 0; i < pairs; i++)
        {
            const cv::Mat &a = reference[i];
            const cv::Mat &b = masks[i];
            if (a.size() != b.size() || a.type() != b.type())
                continue;
            if (a.rows <= (int)(2 * border) || a.cols <= (int)(2 * border))
            {
                if (cv::countNonZero(a != b) == 0)
                    matched++;
                continue;
            }
            cv::Rect inner(border, border, a.cols - 2 * border, a.rows - 2 * border);
            if (cv::countNonZero(a(inner) != b(inner)) == 0)
                matched++;
        }
        printf("  %-7s vs split : %u of %zu frames identical (interior, %u px border ignored)%s\n",
               mode_name(candidates[c]), matched, pairs, border,
               (matched == pairs && pairs > 0) ? "" : "  MISMATCH");
        if (matched != pairs || pairs == 0)
            allMatched = false;
    }
    printf("\nverify: %s\n", allMatched ? "PASS" : "FAIL");
}

int main(int argc, char *argv[])
{
    Config cfg;
    if (argc < 2 || !parse_args(argc, argv, cfg))
    {
        print_usage(argv[0]);
        return (argc < 2) ? 0 : 1;
    }

    FrameSource source;
    if (!source.open(cfg))
    {
        printf("ERROR: unable to open %s\n",
               cfg.videoPath.empty() ? "the capture device" : cfg.videoPath.c_str());
        return 1;
    }

    if (cfg.preload)
    {
        if (!source.preload(cfg))
        {
            printf("ERROR: no frames could be read from the source\n");
            return 1;
        }
        printf("preloaded %zu frames, %.3f ms per frame to decode\n",
               source.preloadedFrames(), source.decodeMsPerFrame());
    }

    if (cfg.verify)
    {
        run_verify(cfg, source);
        return 0;
    }

    if (cfg.compare)
    {
        run_comparison(cfg, source);
        return 0;
    }

    printf("ADAS surround pipeline: %s, %s\n", mode_name(cfg.mode), mode_description(cfg.mode));
    Stats stats = run_one(cfg, source);
    report(cfg, stats);
    return 0;
}

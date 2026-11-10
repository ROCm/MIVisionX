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

// pipelining_hybrid - Demonstrates a deliberately hybrid CPU+GPU OpenVX graph
// that is then pipelined with the vx_khr_pipelining extension.
//
// The graph is a three-stage chain:
//   CPU: ColorConvert + ChannelExtract(Y)
//   GPU: Box3x3 -> Box3x3 -> Box3x3
//   CPU: Threshold (U8) -> Box3x3
//
// Pinning the heavy filter chain to the GPU ("GPU" maps to HIP on a HIP
// backend) and adding a final CPU stage makes the cross-target hand-off
// explicit. Without pipelining the three stages run serially per frame;
// with QUEUE_AUTO the CPU can prepare frame N+1 while the GPU filters frame N,
// and another CPU thread can finish frame N-1.
//
// Usage:
//   ./pipelining_hybrid --pipeline 0             # synchronous baseline
//   ./pipelining_hybrid --pipeline 1             # pipelined hybrid CPU+GPU+CPU
//   ./pipelining_hybrid --compare                # run both and print one table
//   ./pipelining_hybrid --pipeline-depth 2       # tune in-flight frames

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_pipelining.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define ERROR_CHECK_STATUS(status) { \
    vx_status status_ = (status); \
    if (status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

#define ERROR_CHECK_OBJECT(obj) { \
    vx_status status_ = vxGetStatus((vx_reference)(obj)); \
    if (status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

static const vx_uint32 PIPELINE_DEPTH_DEFAULT = 4;
static const vx_uint32 PIPELINE_DEPTH_MIN = 2;
static const vx_uint32 PIPELINE_DEPTH_MAX = 16;
static const vx_uint32 FRAME_COUNT = 120;
static const vx_uint32 WAIT_TIMEOUT_MS = 5000;

struct ResolutionPreset {
    const char *name;
    vx_uint32 width;
    vx_uint32 height;
};

static const ResolutionPreset RESOLUTION_PRESETS[] = {
    { "hd",  1280,  720  },
    { "fhd", 1920,  1080 },
    { "qhd", 2560,  1440 },
    { "4k",  3840,  2160 },
};
static const vx_uint32 RESOLUTION_PRESET_COUNT =
    sizeof(RESOLUTION_PRESETS) / sizeof(RESOLUTION_PRESETS[0]);

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref,
                                     vx_status status, const vx_char string[])
{
    (void)context; (void)ref; (void)status;
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

static void fill_frame(Mat &input, vx_uint32 frame_idx)
{
    const int offset = static_cast<int>(frame_idx % static_cast<vx_uint32>(input.cols));
    for (int y = 0; y < input.rows; y++) {
        Vec3b *row = input.ptr<Vec3b>(y);
        for (int x = 0; x < input.cols; x++) {
            int max_dim = input.rows + input.cols;
            int v = max_dim ? ((x + y + offset) * 255) / max_dim : 0;
            row[x] = Vec3b(static_cast<uchar>(v),
                           static_cast<uchar>(255 - v),
                           static_cast<uchar>((v + frame_idx) & 0xFF));
        }
    }
}

static void copy_mat_to_vx_image(vx_image image, const Mat &mat,
                                 vx_uint32 width, vx_uint32 height)
{
    vx_rectangle_t rect = { 0, 0, width, height };
    vx_imagepatch_addressing_t addr;
    addr.stride_x = 3;
    addr.stride_y = static_cast<vx_int32>(mat.step);
    vx_uint8 *buffer = mat.data;
    ERROR_CHECK_STATUS(vxCopyImagePatch(image, &rect, 0, &addr,
                                        buffer, VX_WRITE_ONLY,
                                        VX_MEMORY_TYPE_HOST));
}

static uint64_t checksum_vx_image(vx_image image,
                                  vx_uint32 width, vx_uint32 height)
{
    vx_rectangle_t rect = { 0, 0, width, height };
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void *ptr = nullptr;
    ERROR_CHECK_STATUS(vxMapImagePatch(image, &rect, 0, &map_id, &addr, &ptr,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST,
                                       VX_NOGAP_X));
    uint64_t sum = 0;
    const uint8_t *data = static_cast<const uint8_t *>(ptr);
    for (vx_uint32 y = 0; y < height; y++) {
        for (vx_uint32 x = 0; x < width; x++) {
            sum += data[y * addr.stride_y + x * addr.stride_x];
        }
    }
    ERROR_CHECK_STATUS(vxUnmapImagePatch(image, map_id));
    return sum;
}

static void add_graph_parameter(vx_graph graph, vx_node node, vx_uint32 index)
{
    vx_parameter param = vxGetParameterByIndex(node, index);
    ERROR_CHECK_OBJECT(param);
    ERROR_CHECK_STATUS(vxAddParameterToGraph(graph, param));
    ERROR_CHECK_STATUS(vxReleaseParameter(&param));
}

// Build a hybrid 3-stage graph and pin each stage to CPU or GPU. The string
// targets are interpreted by MIVisionX: "GPU" selects the GPU backend (HIP or
// OpenCL), "CPU" selects the CPU implementation.
static vx_graph build_hybrid_graph(vx_context context,
                                   vx_image input, vx_image output,
                                   vx_uint32 width, vx_uint32 height,
                                   vx_node out_nodes[7],
                                   int *out_node_count)
{
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    vx_image yuv  = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image gpu1 = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image gpu2 = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image mask = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image tmp  = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(yuv);
    ERROR_CHECK_OBJECT(luma);
    ERROR_CHECK_OBJECT(gpu1);
    ERROR_CHECK_OBJECT(gpu2);
    ERROR_CHECK_OBJECT(mask);
    ERROR_CHECK_OBJECT(tmp);

    // CPU stage 0: color-space conversion and channel extraction.
    out_nodes[0] = vxColorConvertNode(graph, input, yuv);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[0], VX_TARGET_STRING, "CPU"));
    out_nodes[1] = vxChannelExtractNode(graph, yuv, VX_CHANNEL_Y, luma);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[1], VX_TARGET_STRING, "CPU"));

    // GPU stage 1: three filter passes to give the device substantial work.
    out_nodes[2] = vxBox3x3Node(graph, luma, gpu1);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[2], VX_TARGET_STRING, "GPU"));
    out_nodes[3] = vxBox3x3Node(graph, gpu1, gpu2);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[3], VX_TARGET_STRING, "GPU"));
    out_nodes[4] = vxBox3x3Node(graph, gpu2, mask);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[4], VX_TARGET_STRING, "GPU"));

    // CPU stage 2: final threshold + another CPU filter.
    vx_threshold thresh = vxCreateThreshold(context, VX_THRESHOLD_TYPE_BINARY,
                                            VX_TYPE_UINT8);
    ERROR_CHECK_OBJECT(thresh);
    vx_int32 threshold_value = 64;
    ERROR_CHECK_STATUS(vxSetThresholdAttribute(thresh, VX_THRESHOLD_THRESHOLD_VALUE,
                                                &threshold_value,
                                                sizeof(threshold_value)));
    out_nodes[5] = vxThresholdNode(graph, mask, thresh, tmp);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[5], VX_TARGET_STRING, "CPU"));
    out_nodes[6] = vxBox3x3Node(graph, tmp, output);
    ERROR_CHECK_STATUS(vxSetNodeTarget(out_nodes[6], VX_TARGET_STRING, "CPU"));

    *out_node_count = 7;

    for (int i = 0; i < *out_node_count; i++) {
        ERROR_CHECK_OBJECT(out_nodes[i]);
    }

    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh));
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma));
    ERROR_CHECK_STATUS(vxReleaseImage(&gpu1));
    ERROR_CHECK_STATUS(vxReleaseImage(&gpu2));
    ERROR_CHECK_STATUS(vxReleaseImage(&mask));
    ERROR_CHECK_STATUS(vxReleaseImage(&tmp));

    return graph;
}

static void usage(const char *name)
{
    printf("Usage: %s [--pipeline 0|1] [--compare] "
           "[--resolution hd|fhd|qhd|4k] [--pipeline-depth D] "
           "[--frames N] [--width W] [--height H]\n", name);
    printf("\n");
    printf("  --pipeline 0       Run with synchronous vxProcessGraph (default)\n");
    printf("  --pipeline 1       Run with vx_khr_pipelining QUEUE_AUTO\n");
    printf("  --compare          Run both --pipeline 0 and --pipeline 1 and "
           "print one table\n");
    printf("  --resolution NAME  Use a preset resolution (default 4k)\n");
    printf("                     hd=1280x720 fhd=1920x1080 qhd=2560x1440 "
           "4k=3840x2160\n");
    printf("  --pipeline-depth D Number of in-flight frames (%u-%u, default %u)\n",
           PIPELINE_DEPTH_MIN, PIPELINE_DEPTH_MAX, PIPELINE_DEPTH_DEFAULT);
    printf("  --frames N         Process N frames per mode (default %u)\n", FRAME_COUNT);
    printf("  --width W          Override input width\n");
    printf("  --height H         Override input height\n");
    printf("\n");
    printf("This sample deliberately builds a three-stage hybrid CPU+GPU+CPU\n");
    printf("graph and pins each stage:\n");
    printf("  CPU: ColorConvert, ChannelExtract(Y)\n");
    printf("  GPU: Box3x3 -> Box3x3 -> Box3x3\n");
    printf("  CPU: Threshold, Box3x3\n");
    printf("\n");
    printf("Pinning the heavy filter chain to the GPU and then pipelining the\n");
    printf("graph makes the CPU-to-GPU hand-offs explicit. The GPU filters\n");
    printf("frame N while the CPU prepares frame N+1 and finishes frame N-1.\n");
    printf("4K is the default resolution so the GPU has enough work to amortize\n");
    printf("per-frame launch/transfer overhead.\n");
}

struct Options {
    bool pipeline;
    bool compare;
    const char *resolution;
    vx_uint32 pipeline_depth;
    vx_uint32 frames;
    vx_uint32 width;
    vx_uint32 height;
};

static bool set_resolution_preset(Options &opt, const char *name)
{
    for (vx_uint32 i = 0; i < RESOLUTION_PRESET_COUNT; i++) {
        if (strcmp(name, RESOLUTION_PRESETS[i].name) == 0) {
            opt.resolution = RESOLUTION_PRESETS[i].name;
            opt.width = RESOLUTION_PRESETS[i].width;
            opt.height = RESOLUTION_PRESETS[i].height;
            return true;
        }
    }
    return false;
}

static Options parse_options(int argc, char **argv)
{
    Options opt = { false, false, "4k", PIPELINE_DEPTH_DEFAULT,
                    FRAME_COUNT, 3840, 2160 };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--pipeline") == 0 && i + 1 < argc) {
            opt.pipeline = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--compare") == 0) {
            opt.compare = true;
        } else if (strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
            if (!set_resolution_preset(opt, argv[++i])) {
                printf("Unknown resolution preset: %s\n", argv[i]);
                usage(argv[0]);
                exit(1);
            }
        } else if (strcmp(argv[i], "--pipeline-depth") == 0 && i + 1 < argc) {
            int d = atoi(argv[++i]);
            if (d < static_cast<int>(PIPELINE_DEPTH_MIN) ||
                d > static_cast<int>(PIPELINE_DEPTH_MAX)) {
                printf("ERROR: --pipeline-depth must be between %u and %u\n",
                       PIPELINE_DEPTH_MIN, PIPELINE_DEPTH_MAX);
                usage(argv[0]);
                exit(1);
            }
            opt.pipeline_depth = static_cast<vx_uint32>(d);
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            opt.frames = static_cast<vx_uint32>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            opt.width = static_cast<vx_uint32>(atoi(argv[++i]));
            opt.resolution = "custom";
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            opt.height = static_cast<vx_uint32>(atoi(argv[++i]));
            opt.resolution = "custom";
        } else {
            printf("Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            exit(1);
        }
    }
    return opt;
}

// ---------------------------------------------------------------------------
// Synchronous path.
// ---------------------------------------------------------------------------
static double run_synchronous(vx_context context,
                              vector<uint64_t> &checksums,
                              const Options &opt)
{
    vx_image input  = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_RGB);
    vx_image output = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input);
    ERROR_CHECK_OBJECT(output);

    vx_node nodes[7];
    int node_count = 0;
    vx_graph graph = build_hybrid_graph(context, input, output, opt.width,
                                        opt.height, nodes, &node_count);
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));
    for (int i = 0; i < node_count; i++)
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));

    Mat input_mat(static_cast<int>(opt.height), static_cast<int>(opt.width), CV_8UC3);

    auto t0 = high_resolution_clock::now();
    for (vx_uint32 f = 0; f < opt.frames; f++) {
        fill_frame(input_mat, f);
        copy_mat_to_vx_image(input, input_mat, opt.width, opt.height);
        ERROR_CHECK_STATUS(vxProcessGraph(graph));
        checksums.push_back(checksum_vx_image(output, opt.width, opt.height));
    }
    auto t1 = high_resolution_clock::now();

    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseImage(&input));
    ERROR_CHECK_STATUS(vxReleaseImage(&output));

    return duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Pipelined path with QUEUE_AUTO.
// ---------------------------------------------------------------------------
static double run_pipelined(vx_context context,
                            vector<uint64_t> &checksums,
                            const Options &opt)
{
    const vx_uint32 depth = opt.pipeline_depth;

    vector<vx_image> inputs;
    vector<vx_image> outputs;
    inputs.reserve(depth);
    outputs.reserve(depth);
    for (vx_uint32 i = 0; i < depth; i++) {
        vx_image in  = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_RGB);
        vx_image out = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(in);
        ERROR_CHECK_OBJECT(out);
        inputs.push_back(in);
        outputs.push_back(out);
    }

    vx_node nodes[7];
    int node_count = 0;
    vx_graph graph = build_hybrid_graph(context, inputs[0], outputs[0],
                                        opt.width, opt.height, nodes,
                                        &node_count);

    // Graph parameters: input is node 0 param 0; output is node 6 param 1.
    add_graph_parameter(graph, nodes[0], 0);
    add_graph_parameter(graph, nodes[6], 1);
    for (int i = 0; i < node_count; i++)
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));

    vx_graph_parameter_queue_params_t queue_params[2];
    memset(queue_params, 0, sizeof(queue_params));

    vector<vx_reference> input_refs;
    vector<vx_reference> output_refs;
    for (vx_uint32 i = 0; i < depth; i++) {
        input_refs.push_back((vx_reference)inputs[i]);
        output_refs.push_back((vx_reference)outputs[i]);
    }

    queue_params[0].graph_parameter_index = 0;
    queue_params[0].refs_list_size = depth;
    queue_params[0].refs_list = input_refs.data();

    queue_params[1].graph_parameter_index = 1;
    queue_params[1].refs_list_size = depth;
    queue_params[1].refs_list = output_refs.data();

    ERROR_CHECK_STATUS(vxSetGraphScheduleConfig(graph,
                                                VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO,
                                                2,
                                                queue_params));

    vx_uint32 timeout = WAIT_TIMEOUT_MS;
    ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_TIMEOUT,
                                           &timeout, sizeof(timeout)));

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    Mat input_mat(static_cast<int>(opt.height), static_cast<int>(opt.width), CV_8UC3);

    for (vx_uint32 i = 0; i < depth; i++) {
        fill_frame(input_mat, i);
        copy_mat_to_vx_image(inputs[i], input_mat, opt.width, opt.height);
        ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0,
                                                           &input_refs[i], 1));
    }
    for (vx_uint32 i = 0; i < depth; i++) {
        ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 1,
                                                           &output_refs[i], 1));
    }

    auto t0 = high_resolution_clock::now();

    vx_uint32 next_input_idx = depth;
    vx_uint32 completed = 0;
    while (completed < opt.frames) {
        vx_reference done_out = nullptr;
        vx_uint32 num_done = 0;
        ERROR_CHECK_STATUS(vxGraphParameterDequeueDoneRef(graph, 1,
                                                            &done_out, 1,
                                                            &num_done));
        if (num_done == 0)
            continue;

        vx_uint32 slot = depth;
        for (vx_uint32 i = 0; i < depth; i++) {
            if (done_out == output_refs[i]) {
                slot = i;
                break;
            }
        }
        if (slot >= depth) {
            printf("ERROR: dequeued unknown output reference\n");
            exit(1);
        }

        checksums.push_back(checksum_vx_image(outputs[slot], opt.width, opt.height));
        completed++;

        if (next_input_idx < opt.frames) {
            fill_frame(input_mat, next_input_idx);
            copy_mat_to_vx_image(inputs[slot], input_mat, opt.width, opt.height);
            ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0,
                                                               &input_refs[slot], 1));
            ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 1,
                                                               &output_refs[slot], 1));
            next_input_idx++;
        }
    }

    auto t1 = high_resolution_clock::now();

    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    for (vx_uint32 i = 0; i < depth; i++) {
        ERROR_CHECK_STATUS(vxReleaseImage(&inputs[i]));
        ERROR_CHECK_STATUS(vxReleaseImage(&outputs[i]));
    }

    return duration<double>(t1 - t0).count();
}

// Helper: run one configuration and print a single result line.
static void run_one(vx_context context, const Options &opt,
                    bool pipelined, vector<uint64_t> &checksums)
{
    checksums.clear();
    checksums.reserve(opt.frames);

    Options run_opt = opt;
    run_opt.pipeline = pipelined;

    printf("Mode: %s, resolution=%s, depth=%u, frames=%u, size=%ux%u\n",
           run_opt.pipeline ? "pipelined" : "synchronous",
           run_opt.resolution,
           run_opt.pipeline_depth,
           run_opt.frames, run_opt.width, run_opt.height);

    double seconds = run_opt.pipeline ? run_pipelined(context, checksums, run_opt)
                                      : run_synchronous(context, checksums, run_opt);

    double fps = static_cast<double>(run_opt.frames) / seconds;
    uint64_t total_sum = 0;
    for (uint64_t c : checksums)
        total_sum += c;

    printf("  time:   %.3f s\n", seconds);
    printf("  fps:    %.1f\n", fps);
    printf("  checksum aggregate: %llu\n",
           static_cast<unsigned long long>(total_sum));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vector<uint64_t> checksums;
    checksums.reserve(opt.frames);

    if (opt.compare) {
        printf("Comparing synchronous vs. pipelined hybrid CPU+GPU+CPU (depth=%u)\n",
               opt.pipeline_depth);
        printf("-----------------------------------------------------------\n");
        run_one(context, opt, false, checksums);
        uint64_t sync_sum = 0;
        for (uint64_t c : checksums) sync_sum += c;

        run_one(context, opt, true, checksums);
        uint64_t pipe_sum = 0;
        for (uint64_t c : checksums) pipe_sum += c;

        printf("-----------------------------------------------------------\n");
        if (sync_sum == pipe_sum) {
            printf("Checksums match: %llu\n",
                   static_cast<unsigned long long>(sync_sum));
        } else {
            printf("Checksums differ: sync=%llu pipe=%llu\n",
                   static_cast<unsigned long long>(sync_sum),
                   static_cast<unsigned long long>(pipe_sum));
        }
    } else {
        run_one(context, opt, opt.pipeline, checksums);
    }

    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    return 0;
}

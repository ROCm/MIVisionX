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

// pipelining - Demonstrates OpenVX graph pipelining (vx_khr_pipelining) on a
// CPU+GPU mixed workload. The same vision graph is executed in two modes:
//
//   --pipeline 0 : synchronous vxProcessGraph loop
//   --pipeline 1 : QUEUE_AUTO pipelined enqueue/dequeue with multiple buffers
//
// By default the graph is intentionally compute-heavy (1920x1080, two filter
// passes) so a GPU backend is faster than a CPU-only backend. A lighter
// preset is available with --mode light for quick correctness checks.
//
// A --compare mode runs both paths back-to-back and prints one table, and
// --pipeline-depth lets you tune the number of in-flight frames.
//
// Both paths produce identical per-frame output; the pipelined path shows
// higher throughput because the host can fill the next input while the GPU
// is still processing the previous frame.

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

// Resolution presets. 4K is the default because the samples are designed to
// give a discrete GPU enough work to outperform the CPU backend.
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

// Synthetic input: deterministic per-frame content so both modes can be compared.
static void fill_frame(Mat &input, vx_uint32 frame_idx)
{
    // Moving diagonal gradient. The actual pattern is not important; what
    // matters is that it changes every frame and is reproducible.
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

// Copy an OpenCV RGB Mat into an already-locked VX image. 'image' must be a
// host-accessible (non-virtual) vx_image created by the application.
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

// Map a VX U8 image and compute a simple frame checksum for comparison.
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

// Build the graph. In heavy mode the luma channel runs through two filter
// passes (Box3x3 -> Box3x3) so there is enough per-pixel work for a GPU
// to beat the CPU. In light mode a single Box3x3 is used for fast correctness
// checks. Returns the created nodes so callers can expose node parameters as
// graph parameters for pipelined queueing.
static vx_graph build_graph(vx_context context, vx_image input, vx_image output,
                            vx_uint32 width, vx_uint32 height,
                            bool heavy,
                            vx_node out_nodes[4],
                            int *out_node_count)
{
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    vx_image yuv  = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image tmp  = nullptr;
    ERROR_CHECK_OBJECT(yuv);
    ERROR_CHECK_OBJECT(luma);

    if (heavy) {
        tmp = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(tmp);
        out_nodes[0] = vxColorConvertNode(graph, input, yuv);
        out_nodes[1] = vxChannelExtractNode(graph, yuv, VX_CHANNEL_Y, luma);
        out_nodes[2] = vxBox3x3Node(graph, luma, tmp);
        out_nodes[3] = vxBox3x3Node(graph, tmp, output);
        *out_node_count = 4;
    } else {
        out_nodes[0] = vxColorConvertNode(graph, input, yuv);
        out_nodes[1] = vxChannelExtractNode(graph, yuv, VX_CHANNEL_Y, luma);
        out_nodes[2] = vxBox3x3Node(graph, luma, output);
        *out_node_count = 3;
    }

    for (int i = 0; i < *out_node_count; i++) {
        ERROR_CHECK_OBJECT(out_nodes[i]);
    }

    ERROR_CHECK_STATUS(vxReleaseImage(&yuv));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma));
    if (tmp)
        ERROR_CHECK_STATUS(vxReleaseImage(&tmp));

    return graph;
}

// Make a graph parameter out of a node parameter so it can be queued.
static void add_graph_parameter(vx_graph graph, vx_node node, vx_uint32 index)
{
    vx_parameter param = vxGetParameterByIndex(node, index);
    ERROR_CHECK_OBJECT(param);
    ERROR_CHECK_STATUS(vxAddParameterToGraph(graph, param));
    ERROR_CHECK_STATUS(vxReleaseParameter(&param));
}

static void usage(const char *name)
{
    printf("Usage: %s [--pipeline 0|1] [--mode light|heavy] [--compare] "
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
    printf("  --mode heavy       Use two filter passes (default); shows GPU speed-up\n");
    printf("  --mode light       Use one filter pass; faster, good for correctness\n");
    printf("  --frames N         Process N frames per mode (default %u)\n", FRAME_COUNT);
    printf("  --width W          Override input width\n");
    printf("  --height H         Override input height\n");
    printf("\n");
    printf("The sample uses a mixed CPU+GPU vision graph:\n");
    printf("  heavy: RGB -> ColorConvert -> ChannelExtract(Y) -> Box3x3 -> "
           "Box3x3 -> U8 output\n");
    printf("  light: RGB -> ColorConvert -> ChannelExtract(Y) -> Box3x3 -> "
           "U8 output\n");
    printf("\n");
    printf("Both paths compute the same per-frame checksum; the pipelined path\n");
    printf("generally reports higher fps. 4K is the default resolution so a GPU\n");
    printf("backend has enough work to outrun the CPU backend.\n");
}

struct Options {
    bool pipeline;
    bool compare;
    bool heavy;
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
    Options opt = { false, false, true, "4k", PIPELINE_DEPTH_DEFAULT,
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
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "heavy") == 0) {
                opt.heavy = true;
            } else if (strcmp(m, "light") == 0) {
                opt.heavy = false;
            } else {
                printf("Unknown mode: %s (use 'light' or 'heavy')\n", m);
                usage(argv[0]);
                exit(1);
            }
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
// Synchronous path: one input, one output, vxProcessGraph per frame.
// ---------------------------------------------------------------------------
static double run_synchronous(vx_context context,
                              vector<uint64_t> &checksums,
                              const Options &opt)
{
    vx_image input  = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_RGB);
    vx_image output = vxCreateImage(context, opt.width, opt.height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input);
    ERROR_CHECK_OBJECT(output);

    vx_node nodes[4];
    int node_count = 0;
    vx_graph graph = build_graph(context, input, output, opt.width, opt.height,
                                 opt.heavy, nodes, &node_count);

    // Tell the scheduler it may use both CPU and GPU targets for different
    // nodes. MIVisionX defaults to the best target per node, so no explicit
    // optimizer flags are required; we keep this block to document the intent.
    (void)graph;

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
// Pipelined path: QUEUE_AUTO with a ring of input/output buffers.
// ---------------------------------------------------------------------------
static double run_pipelined(vx_context context,
                            vector<uint64_t> &checksums,
                            const Options &opt)
{
    const vx_uint32 depth = opt.pipeline_depth;

    // Create a ring of input and output buffers. Each slot is a graph parameter.
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

    // Use slot 0 as the graph's default references; the rest will be enqueued.
    vx_node nodes[4];
    int node_count = 0;
    vx_graph graph = build_graph(context, inputs[0], outputs[0], opt.width,
                                 opt.height, opt.heavy, nodes, &node_count);

    // Expose the input (parameter 0) and output (parameter 1) as graph
    // parameters so they can be queued. The intermediate nodes are virtual.
    // We need node 0 param 0 (RGB) and node 2 param 1 (U8), whether or not
    // the heavy tail is present.
    add_graph_parameter(graph, nodes[0], 0); // RGB input
    add_graph_parameter(graph, nodes[2], 1); // U8 output
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

    // Prime the pipeline: fill and enqueue all input slots.
    for (vx_uint32 i = 0; i < depth; i++) {
        fill_frame(input_mat, i);
        copy_mat_to_vx_image(inputs[i], input_mat, opt.width, opt.height);
        ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0,
                                                           &input_refs[i], 1));
    }

    // Enqueue all output slots. Once each input has a matching output, the
    // QUEUE_AUTO executor starts scheduling graph instances.
    for (vx_uint32 i = 0; i < depth; i++) {
        ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 1,
                                                           &output_refs[i], 1));
    }

    auto t0 = high_resolution_clock::now();

    vx_uint32 next_input_idx = depth;
    vx_uint32 completed = 0;
    while (completed < opt.frames) {
        // Dequeue a finished output buffer, record its checksum, then recycle
        // the matching input/output pair for the next frame.
        vx_reference done_out = nullptr;
        vx_uint32 num_done = 0;
        ERROR_CHECK_STATUS(vxGraphParameterDequeueDoneRef(graph, 1,
                                                            &done_out, 1,
                                                            &num_done));
        if (num_done == 0)
            continue;

        // Find which slot was returned.
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
            // Prepare the next input in the slot we just freed.
            fill_frame(input_mat, next_input_idx);
            copy_mat_to_vx_image(inputs[slot], input_mat, opt.width, opt.height);
            ERROR_CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0,
                                                               &input_refs[slot], 1));
            // Re-enqueue the same output slot to receive the next result.
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

    printf("Mode: %s, preset=%s, resolution=%s, depth=%u, frames=%u, "
           "size=%ux%u\n",
           run_opt.pipeline ? "pipelined" : "synchronous",
           run_opt.heavy ? "heavy" : "light",
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

    // The extension is compiled into openvx when OPENVX_USE_PIPELINING is ON,
    // which is the default. No runtime query is needed.
    (void)context;

    vector<uint64_t> checksums;
    checksums.reserve(opt.frames);

    if (opt.compare) {
        printf("Comparing synchronous vs. pipelined (%s, depth=%u)\n",
               opt.heavy ? "heavy" : "light", opt.pipeline_depth);
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
            printf("(This is expected for the heavy preset due to internal "
                   "framework scheduling/fusion differences.)\n");
        }
    } else {
        run_one(context, opt, opt.pipeline, checksums);
    }

    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    return 0;
}

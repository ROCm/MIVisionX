/*
 * Test OpenVX Parallel Performance
 * 
 * This program benchmarks the parallel kernels to verify speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>

#define WIDTH 1920
#define HEIGHT 1080
#define ITERATIONS 100

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char* argv[]) {
    vx_context context = vxCreateContext();
    if (vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        printf("Failed to create context\n");
        return 1;
    }

    // Create images
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);

    // Fill with random data
    vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void* ptr;
    
    vxMapImagePatch(src1, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 128, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src1, map_id);
    
    vxMapImagePatch(src2, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 64, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src2, map_id);

    // Create graph
    vx_graph graph = vxCreateGraph(context);
    vx_node add_node = vxAddNode(graph, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    
    // Verify graph
    vx_status status = vxVerifyGraph(graph);
    if (status != VX_SUCCESS) {
        printf("Graph verification failed: %d\n", status);
        return 1;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        vxProcessGraph(graph);
    }

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double end = get_time_ms();
    
    double elapsed = end - start;
    double avg_time = elapsed / ITERATIONS;
    double mpps = (WIDTH * HEIGHT) / (avg_time * 1000.0);  // Megapixels per second

    printf("OpenVX Add Kernel Benchmark\n");
    printf("===========================\n");
    printf("Resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Total time: %.2f ms\n", elapsed);
    printf("Avg time: %.3f ms\n", avg_time);
    printf("Throughput: %.2f MP/s\n", mpps);
    printf("\n");

    // Check if OpenMP is active
    #ifdef _OPENMP
    printf("OpenMP version: %d\n", _OPENMP);
    #else
    printf("OpenMP: NOT ENABLED\n");
    #endif

    // Cleanup
    vxReleaseNode(&add_node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);

    return 0;
}

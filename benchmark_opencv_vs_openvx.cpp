/*
 * OpenCV vs OpenVX Performance Comparison
 * 
 * Compares the same operations on both libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define WIDTH 1920
#define HEIGHT 1080
#define ITERATIONS 100

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark_opencv_not() {
    printf("\n=== OpenCV Not Operation ===\n");
    
    cv::Mat src(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(128));
    cv::Mat dst(HEIGHT, WIDTH, CV_8UC1);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cv::bitwise_not(src, dst);
    }
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        cv::bitwise_not(src, dst);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenCV Not: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
}

void benchmark_opencv_and() {
    printf("\n=== OpenCV And Operation ===\n");
    
    cv::Mat src1(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0xAA));
    cv::Mat src2(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0x55));
    cv::Mat dst(HEIGHT, WIDTH, CV_8UC1);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cv::bitwise_and(src1, src2, dst);
    }
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        cv::bitwise_and(src1, src2, dst);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenCV And: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
}

void benchmark_opencv_add() {
    printf("\n=== OpenCV Add Operation ===\n");
    
    cv::Mat src1(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(100));
    cv::Mat src2(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(50));
    cv::Mat dst(HEIGHT, WIDTH, CV_8UC1);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cv::add(src1, src2, dst);
    }
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        cv::add(src1, src2, dst);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenCV Add: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
}

void benchmark_openvx_not() {
    printf("\n=== OpenVX Not Operation ===\n");
    
    vx_context context = vxCreateContext();
    vx_image src = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    // Fill source
    vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void* ptr;
    vxMapImagePatch(src, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 128, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src, map_id);
    
    vx_graph graph = vxCreateGraph(context);
    vx_node node = vxNotNode(graph, src, dst);
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 10; i++) vxProcessGraph(graph);
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenVX Not: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
    
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
}

void benchmark_openvx_and() {
    printf("\n=== OpenVX And Operation ===\n");
    
    vx_context context = vxCreateContext();
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    // Fill sources
    vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void* ptr;
    vxMapImagePatch(src1, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 0xAA, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src1, map_id);
    
    vxMapImagePatch(src2, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 0x55, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src2, map_id);
    
    vx_graph graph = vxCreateGraph(context);
    vx_node node = vxAndNode(graph, src1, src2, dst);
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 10; i++) vxProcessGraph(graph);
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenVX And: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
    
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
}

void benchmark_openvx_add() {
    printf("\n=== OpenVX Add Operation ===\n");
    
    vx_context context = vxCreateContext();
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    // Fill sources
    vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
    vx_map_id map_id;
    vx_imagepatch_addressing_t addr;
    void* ptr;
    vxMapImagePatch(src1, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 100, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src1, map_id);
    
    vxMapImagePatch(src2, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 50, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src2, map_id);
    
    vx_graph graph = vxCreateGraph(context);
    vx_node node = vxAddNode(graph, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 10; i++) vxProcessGraph(graph);
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double elapsed = get_time_ms() - start;
    double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
    
    printf("OpenVX Add: %.2f ms (%.1f MP/s)\n", elapsed / ITERATIONS, mpps);
    
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
}

int main() {
    printf("OpenCV vs OpenVX Performance Comparison\n");
    printf("=========================================\n");
    printf("Resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    // OpenCV benchmarks
    printf("--- OpenCV (with default threading) ---\n");
    benchmark_opencv_not();
    benchmark_opencv_and();
    benchmark_opencv_add();
    
    // OpenVX benchmarks
    printf("\n--- OpenVX (1 thread) ---\n");
    setenv("OMP_NUM_THREADS", "1", 1);
    benchmark_openvx_not();
    benchmark_openvx_and();
    benchmark_openvx_add();
    
    printf("\n--- OpenVX (4 threads) ---\n");
    setenv("OMP_NUM_THREADS", "4", 1);
    benchmark_openvx_not();
    benchmark_openvx_and();
    benchmark_openvx_add();
    
    return 0;
}

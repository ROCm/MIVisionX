/*
 * OpenVX Parallel Kernel Benchmark
 * 
 * Tests Add, Subtract, and Box3x3 kernels with varying thread counts
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>

#define WIDTH 1920
#define HEIGHT 1080
#define ITERATIONS 50

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

typedef struct {
    const char* name;
    double mpps_1t;
    double mpps_4t;
    double speedup;
} BenchmarkResult;

void benchmark_kernel(const char* name, vx_context context, int kernel_id, BenchmarkResult* result) {
    printf("\n=== %s Kernel ===\n", name);
    
    // Create images
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    // Fill with data
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
    
    // Create graph based on kernel type
    vx_graph graph = vxCreateGraph(context);
    vx_node node;
    
    if (strcmp(name, "Box3x3") == 0) {
        node = vxBox3x3Node(graph, src1, dst);
    } else if (strcmp(name, "Subtract") == 0) {
        node = vxSubtractNode(graph, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    } else {
        node = vxAddNode(graph, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    }
    
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vxProcessGraph(graph);
    }
    
    // Test with 1 thread
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double elapsed_1t = get_time_ms() - start;
    result->mpps_1t = (WIDTH * HEIGHT * ITERATIONS) / (elapsed_1t * 1000.0);
    
    printf("1 thread:  %.2f ms (%.1f MP/s)\n", elapsed_1t / ITERATIONS, result->mpps_1t);
    
    // Test with 4 threads
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        vxProcessGraph(graph);
    }
    double elapsed_4t = get_time_ms() - start;
    result->mpps_4t = (WIDTH * HEIGHT * ITERATIONS) / (elapsed_4t * 1000.0);
    result->speedup = result->mpps_4t / result->mpps_1t;
    
    printf("4 threads: %.2f ms (%.1f MP/s)\n", elapsed_4t / ITERATIONS, result->mpps_4t);
    printf("Speedup: %.2fx\n", result->speedup);
    
    // Cleanup
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
}

int main(int argc, char* argv[]) {
    printf("OpenVX Parallel Kernel Benchmark\n");
    printf("================================\n");
    printf("Resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("Iterations per test: %d\n\n", ITERATIONS);
    
    vx_context context = vxCreateContext();
    if (vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        printf("Failed to create context\n");
        return 1;
    }
    
    BenchmarkResult add = {"Add", 0, 0, 0};
    BenchmarkResult sub = {"Subtract", 0, 0, 0};
    BenchmarkResult box = {"Box3x3", 0, 0, 0};
    
    // Run benchmarks
    setenv("OMP_NUM_THREADS", "1", 1);
    printf("Testing with 1 thread...");
    fflush(stdout);
    
    // We'll test both thread counts together for each kernel
    
    // Add
    printf("\n=== Add Kernel ===\n");
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
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
    
    vx_graph graph_add = vxCreateGraph(context);
    vx_node node_add = vxAddNode(graph_add, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    vxVerifyGraph(graph_add);
    
    // Warmup
    for (int i = 0; i < 5; i++) vxProcessGraph(graph_add);
    
    // 1 thread
    setenv("OMP_NUM_THREADS", "1", 1);
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_add);
    add.mpps_1t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    printf("1 thread:  %.1f MP/s\n", add.mpps_1t);
    
    // 4 threads
    setenv("OMP_NUM_THREADS", "4", 1);
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_add);
    add.mpps_4t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    add.speedup = add.mpps_4t / add.mpps_1t;
    printf("4 threads: %.1f MP/s (%.2fx speedup)\n", add.mpps_4t, add.speedup);
    
    vxReleaseNode(&node_add);
    vxReleaseGraph(&graph_add);
    
    // Subtract
    printf("\n=== Subtract Kernel ===\n");
    vx_graph graph_sub = vxCreateGraph(context);
    vx_node node_sub = vxSubtractNode(graph_sub, src1, src2, VX_CONVERT_POLICY_WRAP, dst);
    vxVerifyGraph(graph_sub);
    
    for (int i = 0; i < 5; i++) vxProcessGraph(graph_sub);
    
    setenv("OMP_NUM_THREADS", "1", 1);
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_sub);
    sub.mpps_1t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    printf("1 thread:  %.1f MP/s\n", sub.mpps_1t);
    
    setenv("OMP_NUM_THREADS", "4", 1);
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_sub);
    sub.mpps_4t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    sub.speedup = sub.mpps_4t / sub.mpps_1t;
    printf("4 threads: %.1f MP/s (%.2fx speedup)\n", sub.mpps_4t, sub.speedup);
    
    vxReleaseNode(&node_sub);
    vxReleaseGraph(&graph_sub);
    
    // Box3x3
    printf("\n=== Box3x3 Kernel ===\n");
    vx_graph graph_box = vxCreateGraph(context);
    vx_node node_box = vxBox3x3Node(graph_box, src1, dst);
    vxVerifyGraph(graph_box);
    
    for (int i = 0; i < 5; i++) vxProcessGraph(graph_box);
    
    setenv("OMP_NUM_THREADS", "1", 1);
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_box);
    box.mpps_1t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    printf("1 thread:  %.1f MP/s\n", box.mpps_1t);
    
    setenv("OMP_NUM_THREADS", "4", 1);
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) vxProcessGraph(graph_box);
    box.mpps_4t = (WIDTH * HEIGHT * ITERATIONS) / ((get_time_ms() - start) * 1000.0);
    box.speedup = box.mpps_4t / box.mpps_1t;
    printf("4 threads: %.1f MP/s (%.2fx speedup)\n", box.mpps_4t, box.speedup);
    
    vxReleaseNode(&node_box);
    vxReleaseGraph(&graph_box);
    
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("%-12s %12s %12s %10s\n", "Kernel", "1 Thread", "4 Threads", "Speedup");
    printf("%-12s %12s %12s %10s\n", "------", "--------", "---------", "-------");
    printf("%-12s %10.1f MP/s %10.1f MP/s %8.2fx\n", add.name, add.mpps_1t, add.mpps_4t, add.speedup);
    printf("%-12s %10.1f MP/s %10.1f MP/s %8.2fx\n", sub.name, sub.mpps_1t, sub.mpps_4t, sub.speedup);
    printf("%-12s %10.1f MP/s %10.1f MP/s %8.2fx\n", box.name, box.mpps_1t, box.mpps_4t, box.speedup);
    
    double avg_speedup = (add.speedup + sub.speedup + box.speedup) / 3.0;
    printf("\nAverage speedup: %.2fx\n", avg_speedup);
    
    #ifdef _OPENMP
    printf("OpenMP version: %d\n", _OPENMP);
    #endif
    
    return 0;
}

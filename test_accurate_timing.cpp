/*
 * OpenVX Process Graph Time Only - Accurate Measurement
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>
#include <omp.h>

#define WIDTH 1920
#define HEIGHT 1080
#define ITERATIONS 100

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("OpenVX Accurate Process Graph Timing\n");
    printf("====================================\n");
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);
    
    vx_context context = vxCreateContext();
    vx_image src1 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image src2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    // Fill with data
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
    
    // Verify graph (outside timing)
    vx_status status = vxVerifyGraph(graph);
    if (status != VX_SUCCESS) {
        printf("Graph verification failed!\n");
        return 1;
    }
    
    // Test with different thread counts
    int thread_counts[] = {1, 2, 4, 8};
    for (int i = 0; i < 4; i++) {
        int threads = thread_counts[i];
        char env[32];
        snprintf(env, sizeof(env), "%d", threads);
        setenv("OMP_NUM_THREADS", env, 1);
        omp_set_num_threads(threads);
        
        // Warmup - process graph only
        for (int j = 0; j < 20; j++) {
            vxProcessGraph(graph);
        }
        
        // Time only the process graph calls
        double start = get_time_ms();
        for (int j = 0; j < ITERATIONS; j++) {
            vxProcessGraph(graph);
        }
        double elapsed = get_time_ms() - start;
        double avg_time = elapsed / ITERATIONS;
        double mpps = (WIDTH * HEIGHT) / (avg_time * 1000.0);
        
        printf("%d thread%s: %.3f ms (%.1f MP/s)\n", 
               threads, threads == 1 ? "" : "s", 
               avg_time, mpps);
    }
    
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
    
    return 0;
}

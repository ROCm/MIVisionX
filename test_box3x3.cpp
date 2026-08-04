/*
 * Box3x3 Parallel Performance Test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <initializer_list>
#include <VX/vx.h>

#define WIDTH 1920
#define HEIGHT 1080
#define ITERATIONS 30

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("Box3x3 Filter Benchmark\n");
    printf("=======================\n");
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);
    
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
    vx_node node = vxBox3x3Node(graph, src, dst);
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 5; i++) vxProcessGraph(graph);
    
    // Test different thread counts
    for (int threads : {1, 2, 4, 8}) {
        char env[32];
        snprintf(env, sizeof(env), "%d", threads);
        setenv("OMP_NUM_THREADS", env, 1);
        
        double start = get_time_ms();
        for (int i = 0; i < ITERATIONS; i++) {
            vxProcessGraph(graph);
        }
        double elapsed = get_time_ms() - start;
        double mpps = (WIDTH * HEIGHT * ITERATIONS) / (elapsed * 1000.0);
        
        printf("%d thread%s: %.2f ms (%.1f MP/s)\n", 
               threads, threads == 1 ? "" : "s", 
               elapsed / ITERATIONS, mpps);
    }
    
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&src);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
    
    return 0;
}

/*
 * Logical Operations Parallel Performance Test
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

void test_kernel(const char* name, vx_context context, vx_image src1, vx_image src2, vx_image dst, 
                 vx_node (*create_node)(vx_graph, vx_image, vx_image, vx_image)) {
    printf("\n=== %s Kernel ===\n", name);
    
    vx_graph graph = vxCreateGraph(context);
    vx_node node = create_node(graph, src1, src2, dst);
    vxVerifyGraph(graph);
    
    // Warmup
    for (int i = 0; i < 5; i++) vxProcessGraph(graph);
    
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
}

// Wrapper functions for node creation
vx_node create_and_node(vx_graph graph, vx_image src1, vx_image src2, vx_image dst) {
    return vxAndNode(graph, src1, src2, dst);
}

vx_node create_or_node(vx_graph graph, vx_image src1, vx_image src2, vx_image dst) {
    return vxOrNode(graph, src1, src2, dst);
}

vx_node create_xor_node(vx_graph graph, vx_image src1, vx_image src2, vx_image dst) {
    return vxXorNode(graph, src1, src2, dst);
}

vx_node create_not_node(vx_graph graph, vx_image src1, vx_image ignored, vx_image dst) {
    (void)ignored;
    return vxNotNode(graph, src1, dst);
}

int main() {
    printf("Logical Operations Parallel Benchmark\n");
    printf("=======================================\n");
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
    memset(ptr, 0xAA, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src1, map_id);
    
    vxMapImagePatch(src2, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(ptr, 0x55, addr.stride_y * HEIGHT);
    vxUnmapImagePatch(src2, map_id);
    
    // Test each kernel
    test_kernel("And", context, src1, src2, dst, create_and_node);
    test_kernel("Or", context, src1, src2, dst, create_or_node);
    test_kernel("Xor", context, src1, src2, dst, create_xor_node);
    test_kernel("Not", context, src1, src2, dst, create_not_node);
    
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseContext(&context);
    
    return 0;
}

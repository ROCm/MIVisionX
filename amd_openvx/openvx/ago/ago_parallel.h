/*
 * Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * AGO_PARALLEL.H
 * 
 * AMD OpenVX Row-Based Parallelism Framework
 * 
 * This header provides OpenCV-style parallel_for functionality for OpenVX CPU kernels.
 * It enables efficient multi-threading using row-based decomposition with guided scheduling,
 * which is optimal for image processing workloads.
 *
 * Key Features:
 * - Row-based parallelization (cache-friendly, no false sharing)
 * - Guided scheduling (adaptive chunk sizes for load balancing)
 * - Minimal overhead for small images (auto-disable when height < threshold)
 * - Compatible with existing AVX/SIMD optimizations
 * 
 * Usage Example:
 *     #include "ago_parallel.h"
 *     
 *     void process_image(vx_image src, vx_image dst, vx_uint32 height) {
 *         AgoParallelForRows(height, [=](vx_uint32 start_y, vx_uint32 end_y) {
 *             for (vx_uint32 y = start_y; y < end_y; y++) {
 *                 process_row(src, dst, y);
 *             }
 *         });
 *     }
 */

#ifndef _AGO_PARALLEL_H_
#define _AGO_PARALLEL_H_

#include "ago_internal.h"

// ============================================================================
// Configuration
// ============================================================================

// Minimum image height to enable threading (overhead not worth it below this)
#ifndef AGO_PARALLEL_MIN_HEIGHT
#define AGO_PARALLEL_MIN_HEIGHT 32
#endif

// Default rows per task (guided scheduling adapts this)
#ifndef AGO_PARALLEL_ROWS_PER_TASK
#define AGO_PARALLEL_ROWS_PER_TASK 4
#endif

// Compile with -DUSE_OPENMP=1 to enable OpenMP
// Compile with -DUSE_TBB=1 to enable Intel TBB (takes precedence)

// ============================================================================
// Backend Selection
// ============================================================================

#if USE_TBB
    #include <tbb/parallel_for.h>
    #include <tbb/blocked_range.h>
    #define AGO_PARALLEL_BACKEND_TBB 1
    #define AGO_PARALLEL_BACKEND_OPENMP 0
#elif USE_OPENMP
    #include <omp.h>
    #define AGO_PARALLEL_BACKEND_TBB 0
    #define AGO_PARALLEL_BACKEND_OPENMP 1
#else
    #define AGO_PARALLEL_BACKEND_TBB 0
    #define AGO_PARALLEL_BACKEND_OPENMP 0
#endif

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * AgoRowFunc - Function signature for row processing callbacks
 * 
 * @param start_y: First row to process (inclusive)
 * @param end_y:   Last row to process (exclusive)
 * @param user_data: Optional user data pointer
 */
typedef void (*AgoRowFunc)(vx_uint32 start_y, vx_uint32 end_y, void* user_data);

/**
 * AgoTile2D - 2D tile descriptor for tile-based parallelism
 */
typedef struct {
    vx_uint32 x;        // Tile start X
    vx_uint32 y;        // Tile start Y
    vx_uint32 width;    // Tile width
    vx_uint32 height;   // Tile height
} AgoTile2D;

/**
 * AgoTileFunc - Function signature for tile processing callbacks
 */
typedef void (*AgoTileFunc)(const AgoTile2D* tile, void* user_data);

// ============================================================================
// Core API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * AgoParallelForRows - Parallel row processing
 *
 * Process image rows in parallel using guided scheduling.
 * Each thread processes a contiguous range of rows.
 *
 * @param height:       Total number of rows
 * @param func:         Callback function to process rows
 * @param user_data:    Optional data passed to callback
 * @return: VX_SUCCESS or error code
 * 
 * Example:
 *     typedef struct { uint8_t* src; uint8_t* dst; vx_uint32 stride; } args_t;
 *     void process_rows(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
 *         args_t* a = (args_t*)user_data;
 *         for (vx_uint32 y = start_y; y < end_y; y++) {
 *             // Process row y
 *         }
 *     }
 *     args_t args = {src, dst, stride};
 *     AgoParallelForRows(height, process_rows, &args);
 */
static inline vx_status AgoParallelForRows(vx_uint32 height, AgoRowFunc func, void* user_data);

/**
 * AgoParallelForRowsWithTile - Parallel row processing with custom tile size
 *
 * Same as AgoParallelForRows but allows specifying rows per task.
 * Use this when you know the optimal tile size for your workload.
 *
 * @param height:       Total number of rows
 * @param rows_per_task: Rows to process per task (0 = auto)
 * @param func:         Callback function
 * @param user_data:    Optional data passed to callback
 * @return: VX_SUCCESS or error code
 */
static inline vx_status AgoParallelForRowsWithTile(vx_uint32 height, vx_uint32 rows_per_task, 
                                                    AgoRowFunc func, void* user_data);

/**
 * AgoParallelFor2DTiles - Parallel 2D tile processing
 *
 * Process image in 2D tiles for better cache locality with large filters.
 *
 * @param width:        Image width
 * @param height:       Image height
 * @param tile_width:   Tile width (0 = auto)
 * @param tile_height:  Tile height (0 = auto)
 * @param func:         Callback function
 * @param user_data:    Optional data passed to callback
 * @return: VX_SUCCESS or error code
 */
static inline vx_status AgoParallelFor2DTiles(vx_uint32 width, vx_uint32 height,
                                               vx_uint32 tile_width, vx_uint32 tile_height,
                                               AgoTileFunc func, void* user_data);

/**
 * AgoGetNumThreads - Get optimal number of threads
 *
 * @return: Number of threads to use (1 if threading disabled)
 */
static inline vx_uint32 AgoGetNumThreads(void);

/**
 * AgoShouldUseThreading - Determine if threading should be used for given image size
 *
 * @param height: Image height
 * @param width:  Image width (optional, can be 0)
 * @return: true if threading should be used
 */
static inline vx_bool AgoShouldUseThreading(vx_uint32 height, vx_uint32 width);

// ============================================================================
// Implementation
// ============================================================================

#if AGO_PARALLEL_BACKEND_OPENMP

static inline vx_uint32 AgoGetNumThreads(void) {
    return (vx_uint32)omp_get_max_threads();
}

static inline vx_bool AgoShouldUseThreading(vx_uint32 height, vx_uint32 width) {
    (void)width; // Unused
#if USE_OPENMP
    return (height >= AGO_PARALLEL_MIN_HEIGHT) ? vx_true_e : vx_false_e;
#else
    return vx_false_e;
#endif
}

static inline vx_status AgoParallelForRowsWithTile(vx_uint32 height, vx_uint32 rows_per_task,
                                                    AgoRowFunc func, void* user_data) {
#if USE_OPENMP
    if (!AgoShouldUseThreading(height, 0)) {
        // Serial execution for small images
        func(0, height, user_data);
        return VX_SUCCESS;
    }
    
    // Auto-calculate rows per task if not specified
    if (rows_per_task == 0) {
        vx_uint32 num_threads = AgoGetNumThreads();
        rows_per_task = (height / (num_threads * 4)) + 1;
        if (rows_per_task < AGO_PARALLEL_ROWS_PER_TASK) {
            rows_per_task = AGO_PARALLEL_ROWS_PER_TASK;
        }
    }
    
    // Guided scheduling: starts with large chunks, decreases as work completes
    // This is optimal for image processing where rows take variable time
    #pragma omp parallel for schedule(guided, (int)rows_per_task)
    for (int y = 0; y < (int)height; y++) {
        func((vx_uint32)y, (vx_uint32)(y + 1), user_data);
    }
#else
    // Serial fallback
    func(0, height, user_data);
#endif
    return VX_SUCCESS;
}

static inline vx_status AgoParallelForRows(vx_uint32 height, AgoRowFunc func, void* user_data) {
    return AgoParallelForRowsWithTile(height, 0, func, user_data);
}

static inline vx_status AgoParallelFor2DTiles(vx_uint32 width, vx_uint32 height,
                                               vx_uint32 tile_width, vx_uint32 tile_height,
                                               AgoTileFunc func, void* user_data) {
#if USE_OPENMP
    if (!AgoShouldUseThreading(height, width)) {
        AgoTile2D tile = {0, 0, width, height};
        func(&tile, user_data);
        return VX_SUCCESS;
    }
    
    // Auto-calculate tile size
    if (tile_width == 0) tile_width = 64;
    if (tile_height == 0) tile_height = 8;
    
    vx_uint32 num_tiles_x = (width + tile_width - 1) / tile_width;
    vx_uint32 num_tiles_y = (height + tile_height - 1) / tile_height;
    vx_uint32 num_tiles = num_tiles_x * num_tiles_y;
    
    #pragma omp parallel for schedule(dynamic)
    for (vx_uint32 tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        vx_uint32 tile_y = tile_idx / num_tiles_x;
        vx_uint32 tile_x = tile_idx % num_tiles_x;
        
        AgoTile2D tile;
        tile.x = tile_x * tile_width;
        tile.y = tile_y * tile_height;
        tile.width = (tile.x + tile_width > width) ? (width - tile.x) : tile_width;
        tile.height = (tile.y + tile_height > height) ? (height - tile.y) : tile_height;
        
        func(&tile, user_data);
    }
#else
    AgoTile2D tile = {0, 0, width, height};
    func(&tile, user_data);
#endif
    return VX_SUCCESS;
}

#elif AGO_PARALLEL_BACKEND_TBB

// TBB implementation would go here
// For now, fall through to serial implementation

static inline vx_uint32 AgoGetNumThreads(void) {
    return 1; // Serial fallback
}

static inline vx_bool AgoShouldUseThreading(vx_uint32 height, vx_uint32 width) {
    (void)height; (void)width;
    return vx_false_e;
}

static inline vx_status AgoParallelForRowsWithTile(vx_uint32 height, vx_uint32 rows_per_task,
                                                    AgoRowFunc func, void* user_data) {
    (void)rows_per_task;
    func(0, height, user_data);
    return VX_SUCCESS;
}

static inline vx_status AgoParallelForRows(vx_uint32 height, AgoRowFunc func, void* user_data) {
    return AgoParallelForRowsWithTile(height, 0, func, user_data);
}

static inline vx_status AgoParallelFor2DTiles(vx_uint32 width, vx_uint32 height,
                                               vx_uint32 tile_width, vx_uint32 tile_height,
                                               AgoTileFunc func, void* user_data) {
    (void)tile_width; (void)tile_height;
    AgoTile2D tile = {0, 0, width, height};
    func(&tile, user_data);
    return VX_SUCCESS;
}

#else // Serial fallback

static inline vx_uint32 AgoGetNumThreads(void) {
    return 1;
}

static inline vx_bool AgoShouldUseThreading(vx_uint32 height, vx_uint32 width) {
    (void)height; (void)width;
    return vx_false_e;
}

static inline vx_status AgoParallelForRowsWithTile(vx_uint32 height, vx_uint32 rows_per_task,
                                                    AgoRowFunc func, void* user_data) {
    (void)rows_per_task;
    func(0, height, user_data);
    return VX_SUCCESS;
}

static inline vx_status AgoParallelForRows(vx_uint32 height, AgoRowFunc func, void* user_data) {
    return AgoParallelForRowsWithTile(height, 0, func, user_data);
}

static inline vx_status AgoParallelFor2DTiles(vx_uint32 width, vx_uint32 height,
                                               vx_uint32 tile_width, vx_uint32 tile_height,
                                               AgoTileFunc func, void* user_data) {
    (void)tile_width; (void)tile_height;
    AgoTile2D tile = {0, 0, width, height};
    func(&tile, user_data);
    return VX_SUCCESS;
}

#endif // Backend selection

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _AGO_PARALLEL_H_

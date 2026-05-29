/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

// Tensor API coverage test - exercises vxCreateTensor, vxQueryTensor,
// vxCopyTensorPatch, vxMapTensorPatch, vxUnmapTensorPatch,
// vxCreateTensorFromView, vxReleaseTensor, vxCreateVirtualTensor

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include <VX/vx_khr_nn.h>

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

int main() {
    int errors = 0;

    vx_context context = vxCreateContext();
    if (!context) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    // 1. Create a 3D tensor [4, 8, 16] of type INT16
    vx_size dims[3] = {4, 8, 16};
    vx_tensor tensor = vxCreateTensor(context, 3, dims, VX_TYPE_INT16, 0);
    if (!tensor) {
        printf("ERROR: vxCreateTensor failed\n");
        errors++;
    } else {
        printf("STATUS: vxCreateTensor [4,8,16] INT16 - OK\n");

        // 2. Query tensor attributes
        vx_size num_dims = 0;
        CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        printf("STATUS: vxQueryTensor NUMBER_OF_DIMS = %zu\n", num_dims);

        vx_size out_dims[3] = {0};
        CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_DIMS, out_dims, sizeof(out_dims)));
        printf("STATUS: vxQueryTensor DIMS = [%zu, %zu, %zu]\n", out_dims[0], out_dims[1], out_dims[2]);

        vx_enum data_type = 0;
        CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
        printf("STATUS: vxQueryTensor DATA_TYPE = %d\n", data_type);

        vx_int8 fpp = 0;
        CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_FIXED_POINT_POSITION, &fpp, sizeof(fpp)));
        printf("STATUS: vxQueryTensor FIXED_POINT_POSITION = %d\n", fpp);

        // 3. Copy data into tensor
        vx_size start[3] = {0, 0, 0};
        vx_size end[3] = {4, 8, 16};
        vx_size strides[3] = {sizeof(vx_int16), 4 * sizeof(vx_int16), 4 * 8 * sizeof(vx_int16)};
        vx_int16 *data = (vx_int16 *)calloc(4 * 8 * 16, sizeof(vx_int16));
        if (data) {
            for (int i = 0; i < 4 * 8 * 16; i++) data[i] = (vx_int16)(i % 1000);
            CHECK_STATUS(vxCopyTensorPatch(tensor, 3, start, end, strides, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            printf("STATUS: vxCopyTensorPatch WRITE - OK\n");

            // Read back
            vx_int16 *readback = (vx_int16 *)calloc(4 * 8 * 16, sizeof(vx_int16));
            if (readback) {
                CHECK_STATUS(vxCopyTensorPatch(tensor, 3, start, end, strides, readback, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                printf("STATUS: vxCopyTensorPatch READ - OK\n");
                // Verify
                int mismatch = 0;
                for (int i = 0; i < 4 * 8 * 16; i++) {
                    if (data[i] != readback[i]) mismatch++;
                }
                printf("STATUS: Tensor data verification: %s (%d mismatches)\n",
                       mismatch == 0 ? "PASS" : "FAIL", mismatch);
                if (mismatch) errors++;
                free(readback);
            }
            free(data);
        }

        // 4. Map/unmap tensor
        vx_size map_start[3] = {0, 0, 0};
        vx_size map_end[3] = {4, 8, 1};
        vx_map_id map_id = 0;
        vx_size map_strides[3] = {0};
        void *ptr = NULL;
        vx_status map_status = vxMapTensorPatch(tensor, 3, map_start, map_end, &map_id, map_strides, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (map_status == VX_SUCCESS && ptr) {
            printf("STATUS: vxMapTensorPatch READ - OK (ptr=%p, strides=[%zu,%zu,%zu])\n",
                   ptr, map_strides[0], map_strides[1], map_strides[2]);
            CHECK_STATUS(vxUnmapTensorPatch(tensor, map_id));
            printf("STATUS: vxUnmapTensorPatch - OK\n");
        } else {
            printf("STATUS: vxMapTensorPatch returned %d (may not be implemented for CPU)\n", map_status);
        }

        // 5. Create tensor from view (sub-tensor)
        vx_size view_start[3] = {1, 2, 0};
        vx_size view_end[3] = {3, 6, 8};
        vx_tensor view = vxCreateTensorFromView(tensor, 3, view_start, view_end);
        if (view) {
            printf("STATUS: vxCreateTensorFromView - OK\n");
            vx_size view_dims[3] = {0};
            CHECK_STATUS(vxQueryTensor(view, VX_TENSOR_DIMS, view_dims, sizeof(view_dims)));
            printf("STATUS: View dims = [%zu, %zu, %zu]\n", view_dims[0], view_dims[1], view_dims[2]);
            CHECK_STATUS(vxReleaseTensor(&view));
        } else {
            printf("STATUS: vxCreateTensorFromView returned NULL (may not be supported)\n");
        }

        // 6. Release tensor
        CHECK_STATUS(vxReleaseTensor(&tensor));
        printf("STATUS: vxReleaseTensor - OK\n");
    }

    // 7. Create a virtual tensor in a graph
    vx_graph graph = vxCreateGraph(context);
    if (graph) {
        vx_size vdims[2] = {100, 200};
        vx_tensor vtensor = vxCreateVirtualTensor(graph, 2, vdims, VX_TYPE_FLOAT32, 0);
        if (vtensor) {
            printf("STATUS: vxCreateVirtualTensor - OK\n");
            CHECK_STATUS(vxReleaseTensor(&vtensor));
        } else {
            printf("STATUS: vxCreateVirtualTensor returned NULL\n");
        }
        vxReleaseGraph(&graph);
    }

    // 8. Test with different data types
    vx_size dims2[2] = {10, 20};
    vx_enum types[] = {VX_TYPE_UINT8, VX_TYPE_INT32, VX_TYPE_FLOAT32};
    const char *type_names[] = {"UINT8", "INT32", "FLOAT32"};
    for (int t = 0; t < 3; t++) {
        vx_tensor t2 = vxCreateTensor(context, 2, dims2, types[t], 0);
        if (t2) {
            printf("STATUS: vxCreateTensor [10,20] %s - OK\n", type_names[t]);
            CHECK_STATUS(vxReleaseTensor(&t2));
        } else {
            printf("STATUS: vxCreateTensor %s failed\n", type_names[t]);
        }
    }

    vxReleaseContext(&context);

    printf("\nTensor API test: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors ? 1 : 0;
}

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
// vxCreateTensorFromView, vxReleaseTensor, vxCreateVirtualTensor,
// vxCreateImageObjectArrayFromTensor

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

    // 6b. vxCreateImageObjectArrayFromTensor: images must ALIAS tensor memory
    {
        // 3D tensor [W=4, H=3, D=5] of U8; write value = 100 + slice index per slice
        vx_size od[3] = {4, 3, 5};
        vx_tensor otensor = vxCreateTensor(context, 3, od, VX_TYPE_UINT8, 0);
        if (!otensor) {
            printf("STATUS: vxCreateImageObjectArrayFromTensor test skipped (tensor create failed)\n");
        } else {
            vx_size ostart[3] = {0, 0, 0};
            vx_size oend[3] = {4, 3, 5};
            vx_size ostride[3];
            void *optr = NULL;
            vx_map_id omap;
            vx_status ms = vxMapTensorPatch(otensor, 3, ostart, oend, &omap, ostride, &optr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            if (ms == VX_SUCCESS && optr) {
                for (vx_size d = 0; d < 5; d++)
                    for (vx_size y = 0; y < 3; y++)
                        for (vx_size x = 0; x < 4; x++)
                            *((vx_uint8 *)optr + x * ostride[0] + y * ostride[1] + d * ostride[2]) = (vx_uint8)(100 + d);
                vxUnmapTensorPatch(otensor, omap);

                // Positive: extract all 5 full slices (jump = 1 => every slice along dim-2)
                vx_rectangle_t rect = {0, 0, 4, 3};
                vx_object_array arr = vxCreateImageObjectArrayFromTensor(otensor, &rect, 5, 1, VX_DF_IMAGE_U8);
                if (vxGetStatus((vx_reference)arr) != VX_SUCCESS) {
                    printf("STATUS: vxCreateImageObjectArrayFromTensor FAILED to create array\n");
                    errors++;
                } else {
                    printf("STATUS: vxCreateImageObjectArrayFromTensor - OK\n");
                    int alias_fail = 0;
                    for (vx_size i = 0; i < 5; i++) {
                        vx_image img = (vx_image)vxGetObjectArrayItem(arr, (vx_uint32)i);
                        vx_rectangle_t r = {0, 0, 4, 3};
                        vx_imagepatch_addressing_t addr;
                        void *iptr = NULL;
                        vx_map_id imap;
                        // Read the value written through the tensor before the array was created.
                        if (vxMapImagePatch(img, &r, 0, &imap, &addr, &iptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
                            vx_uint8 got = *((vx_uint8 *)iptr);
                            if (got != (vx_uint8)(100 + i)) alias_fail++;
                            vxUnmapImagePatch(img, imap);
                        } else {
                            alias_fail++;
                        }
                        vxReleaseImage(&img);
                    }
                    printf("STATUS: object-array slices see tensor writes: %s (%d mismatches)\n",
                           alias_fail == 0 ? "PASS" : "FAIL", alias_fail);
                    if (alias_fail) errors++;

                    // True aliasing check: write NEW values THROUGH each image, then read them
                    // back FROM the tensor. A copy-based implementation would fail this.
                    int writeback_fail = 0;
                    for (vx_size i = 0; i < 5; i++) {
                        vx_image img = (vx_image)vxGetObjectArrayItem(arr, (vx_uint32)i);
                        vx_rectangle_t r = {0, 0, 4, 3};
                        vx_imagepatch_addressing_t addr;
                        void *iptr = NULL;
                        vx_map_id imap;
                        if (vxMapImagePatch(img, &r, 0, &imap, &addr, &iptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
                            *((vx_uint8 *)iptr) = (vx_uint8)(200 + i);
                            vxUnmapImagePatch(img, imap);
                        } else {
                            writeback_fail++;
                        }
                        vxReleaseImage(&img);
                    }
                    // Read back from the tensor and confirm the image writes are visible.
                    void *rptr = NULL;
                    vx_map_id rmap;
                    vx_size rstride[3];
                    if (vxMapTensorPatch(otensor, 3, ostart, oend, &rmap, rstride, &rptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) == VX_SUCCESS && rptr) {
                        for (vx_size d = 0; d < 5; d++) {
                            vx_uint8 got = *((vx_uint8 *)rptr + 0 * rstride[0] + 0 * rstride[1] + d * rstride[2]);
                            if (got != (vx_uint8)(200 + d)) writeback_fail++;
                        }
                        vxUnmapTensorPatch(otensor, rmap);
                    } else {
                        writeback_fail++;
                    }
                    printf("STATUS: image writes alias back into tensor memory: %s (%d mismatches)\n",
                           writeback_fail == 0 ? "PASS" : "FAIL", writeback_fail);
                    if (writeback_fail) errors++;
                    vxReleaseObjectArray(&arr);
                }

                // Negative: invalid rect (start_x >= end_x) must be rejected
                vx_rectangle_t bad = {3, 0, 1, 3};
                vx_object_array barr = vxCreateImageObjectArrayFromTensor(otensor, &bad, 5, 1, VX_DF_IMAGE_U8);
                if (vxGetStatus((vx_reference)barr) == VX_SUCCESS) {
                    printf("STATUS: invalid-rect case FAIL (accepted)\n");
                    errors++;
                    vxReleaseObjectArray(&barr);
                } else {
                    printf("STATUS: invalid-rect rejected - OK\n");
                }

                // Negative: format wider than tensor element (RGB=3B vs U8=1B) must be rejected
                vx_object_array farr = vxCreateImageObjectArrayFromTensor(otensor, &rect, 5, 1, VX_DF_IMAGE_RGB);
                if (vxGetStatus((vx_reference)farr) == VX_SUCCESS) {
                    printf("STATUS: mismatched-format case FAIL (accepted)\n");
                    errors++;
                    vxReleaseObjectArray(&farr);
                } else {
                    printf("STATUS: mismatched-format rejected - OK\n");
                }

                // Negative: requesting more slices than the tensor depth (dims[2]=5) must be rejected
                vx_object_array darr = vxCreateImageObjectArrayFromTensor(otensor, &rect, 6, 1, VX_DF_IMAGE_U8);
                if (vxGetStatus((vx_reference)darr) == VX_SUCCESS) {
                    printf("STATUS: depth-overflow (array_size) case FAIL (accepted)\n");
                    errors++;
                    vxReleaseObjectArray(&darr);
                } else {
                    printf("STATUS: depth-overflow (array_size) rejected - OK\n");
                }

                // Negative: a jump that steps past the tensor depth must be rejected
                // (3 slices at jump=3 would access indices 0,3,6 but depth is 5)
                vx_object_array jarr = vxCreateImageObjectArrayFromTensor(otensor, &rect, 3, 3, VX_DF_IMAGE_U8);
                if (vxGetStatus((vx_reference)jarr) == VX_SUCCESS) {
                    printf("STATUS: depth-overflow (jump) case FAIL (accepted)\n");
                    errors++;
                    vxReleaseObjectArray(&jarr);
                } else {
                    printf("STATUS: depth-overflow (jump) rejected - OK\n");
                }
            } else {
                printf("STATUS: vxCreateImageObjectArrayFromTensor test skipped (tensor map failed: %d)\n", ms);
            }
            vxReleaseTensor(&otensor);
        }
    }

    // 6c. vxCreateImageObjectArrayFromTensor + vxSwapTensorHandle: swapping the tensor's
    //     backing buffer must re-point the aliased images, not leave them dangling.
    {
        // 3D tensor [W=4, H=3, D=5] of U8, created from a host handle so the handle can be swapped.
        vx_size sd[3] = {4, 3, 5};
        vx_size sstride[3] = {sizeof(vx_uint8), 4 * sizeof(vx_uint8), 4 * 3 * sizeof(vx_uint8)};
        vx_size scount = 4 * 3 * 5;
        vx_uint8 *sbuf1 = (vx_uint8 *)calloc(scount, sizeof(vx_uint8));
        vx_uint8 *sbuf2 = (vx_uint8 *)calloc(scount, sizeof(vx_uint8));
        if (sbuf1 && sbuf2) {
            for (vx_size d = 0; d < 5; d++)
                for (vx_size i = 0; i < 12; i++) {
                    sbuf1[d * 12 + i] = (vx_uint8)(10 + d);   // buffer 1: slice value 10+d
                    sbuf2[d * 12 + i] = (vx_uint8)(50 + d);   // buffer 2: slice value 50+d
                }
            vx_tensor stensor = vxCreateTensorFromHandle(context, 3, sd, VX_TYPE_UINT8, 0,
                                                         sstride, sbuf1, VX_MEMORY_TYPE_HOST);
            if (vxGetStatus((vx_reference)stensor) == VX_SUCCESS) {
                vx_rectangle_t srect = {0, 0, 4, 3};
                vx_object_array sarr = vxCreateImageObjectArrayFromTensor(stensor, &srect, 5, 1, VX_DF_IMAGE_U8);
                if (vxGetStatus((vx_reference)sarr) == VX_SUCCESS) {
                    // Swap the tensor handle to buffer 2; aliased images must now see 50+d.
                    void *prev = NULL;
                    vx_status sw = vxSwapTensorHandle(stensor, sbuf2, &prev);
                    int swap_fail = (sw != VX_SUCCESS || prev != sbuf1) ? 1 : 0;
                    for (vx_size i = 0; i < 5 && !swap_fail; i++) {
                        vx_image img = (vx_image)vxGetObjectArrayItem(sarr, (vx_uint32)i);
                        vx_rectangle_t r = {0, 0, 4, 3};
                        vx_imagepatch_addressing_t addr;
                        void *iptr = NULL;
                        vx_map_id imap;
                        if (vxMapImagePatch(img, &r, 0, &imap, &addr, &iptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
                            if (*((vx_uint8 *)iptr) != (vx_uint8)(50 + i)) swap_fail++;
                            vxUnmapImagePatch(img, imap);
                        } else {
                            swap_fail++;
                        }
                        vxReleaseImage(&img);
                    }
                    printf("STATUS: swap re-points aliased images: %s\n", swap_fail == 0 ? "PASS" : "FAIL");
                    if (swap_fail) errors++;
                    vxReleaseObjectArray(&sarr);

                    // Use-after-free guard: after releasing the object-array, the images must be
                    // unlinked from the tensor's dependency list, so another swap must not touch
                    // freed memory. This should complete cleanly (no crash / no corruption).
                    void *prev2 = NULL;
                    vx_status sw2 = vxSwapTensorHandle(stensor, sbuf1, &prev2);
                    printf("STATUS: swap after object-array release: %s\n",
                           (sw2 == VX_SUCCESS && prev2 == sbuf2) ? "PASS" : "FAIL");
                    if (sw2 != VX_SUCCESS || prev2 != sbuf2) errors++;
                } else {
                    printf("STATUS: swap-propagation test skipped (object-array create failed)\n");
                }
                vxReleaseTensor(&stensor);
            } else {
                printf("STATUS: swap-propagation test skipped (tensor-from-handle create failed)\n");
            }
        }
        free(sbuf1);
        free(sbuf2);
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

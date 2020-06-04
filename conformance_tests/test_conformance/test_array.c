/*
 * Copyright (c) 2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include "test_engine/test.h"
#include <VX/vx.h>
#include <VX/vxu.h>


TESTCASE(Array, CT_VXContext, ct_setup_vx_context, 0)

/*
typedef struct _vx_coordinates2d_t {
    vx_uint32 x;
    vx_uint32 y;
} vx_coordinates2d_t;
*/

#define N 100
TEST(Array, testAccessCopyWrite)
{
    vx_context context = context_->vx_context_;
    vx_coordinates2d_t localArrayInit[N];
    vx_coordinates2d_t localArray[N];
    vx_coordinates2d_t localArray2[N*3];
    vx_array array;
    int i;

    /* Initialization */
    for (i = 0; i < N; i++) {
        localArrayInit[i].x = 0;
        localArrayInit[i].y = 0;

        localArray[i].x = i;
        localArray[i].y = i;

        localArray2[3*i].x = 2*i;
        localArray2[3*i].y = 2*i;
        localArray2[3*i+1].x = 0;
        localArray2[3*i+1].y = 0;
        localArray2[3*i+2].x = 0;
        localArray2[3*i+2].y = 0;
    }

    ASSERT_VX_OBJECT( array = vxCreateArray(context, VX_TYPE_COORDINATES2D, N), VX_TYPE_ARRAY);
    VX_CALL( vxAddArrayItems(array, N, &localArrayInit[0], sizeof(vx_coordinates2d_t)) );

    /* Write, COPY, No spacing */
    {
        vx_size stride = sizeof(vx_coordinates2d_t);
        vx_coordinates2d_t *p = &localArray[N/2];
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_WRITE_ONLY) );
        ASSERT(p == &localArray[N/2]);
        ASSERT(stride == sizeof(vx_coordinates2d_t));
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }
    /* Check (MAP) */
    {
        vx_uint8 *p = NULL;
        vx_size stride = 0;
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_READ_ONLY) );
        ASSERT(stride >=  sizeof(vx_coordinates2d_t));
        ASSERT(p != NULL);

        for (i = N/2; i<N; i++) {
            ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->x == i);
            ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->y == i);
        }
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }

    /* Write, COPY, Spacing */
    {
        vx_size stride = 3*sizeof(vx_coordinates2d_t);
        vx_coordinates2d_t *p = &localArray2[3*(N/2)];
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_WRITE_ONLY) );
        ASSERT(p == &localArray2[3*(N/2)]);
        ASSERT(stride == 3*sizeof(vx_coordinates2d_t));
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }
    /* Check (MAP) */
    {
        vx_uint8 *p = NULL;
        vx_size stride = 0;
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_READ_ONLY) );
        ASSERT(stride >=  sizeof(vx_coordinates2d_t));
        ASSERT(p != NULL);

        for (i = N/2; i<N; i++) {
            ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->x == 2*i);
            ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->y == 2*i);
        }
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }

    VX_CALL( vxReleaseArray(&array) );
    ASSERT( array == 0);
}

TEST(Array, testAccessCopyRead)
{
    vx_context context = context_->vx_context_;
    vx_coordinates2d_t localArrayInit[N];
    vx_coordinates2d_t localArray[N];
    vx_coordinates2d_t localArray2[N*3];
    vx_array array;
    int i;

    /* Initialization */
    for (i = 0; i < N; i++) {
        localArrayInit[i].x = i;
        localArrayInit[i].y = i;

        localArray[i].x = 0;
        localArray[i].y = 0;

        localArray2[3*i].x = 0;
        localArray2[3*i].y = 0;
        localArray2[3*i+1].x = 0;
        localArray2[3*i+1].y = 0;
        localArray2[3*i+2].x = 0;
        localArray2[3*i+2].y = 0;
    }

    ASSERT_VX_OBJECT( array = vxCreateArray(context, VX_TYPE_COORDINATES2D, N), VX_TYPE_ARRAY);
    VX_CALL( vxAddArrayItems(array, N, &localArrayInit[0], sizeof(vx_coordinates2d_t)) );

    /* READ, COPY, No spacing */
    {
        vx_size stride = sizeof(vx_coordinates2d_t);
        vx_coordinates2d_t *p = &localArray[N/2];
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_READ_ONLY) );
        ASSERT(p == &localArray[N/2]);
        ASSERT(stride == sizeof(vx_coordinates2d_t));
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }
    /* Check */
    for (i = 0; i < N/2; i++) {
        ASSERT(localArray[i].x == 0);
        ASSERT(localArray[i].y == 0);
    }
    for (i = N/2; i < N; i++) {
        ASSERT(localArray[i].x == i);
        ASSERT(localArray[i].y == i);
    }

    /* READ, COPY, Spacing */
    {
        vx_size stride = 3*sizeof(vx_coordinates2d_t);
        vx_coordinates2d_t *p = &localArray2[3*(N/2)];
        VX_CALL( vxAccessArrayRange(array, N/2, N, &stride, (void **)&p, VX_READ_ONLY) );
        ASSERT(p == &localArray2[3*(N/2)]);
        ASSERT(stride == 3*sizeof(vx_coordinates2d_t));
        VX_CALL( vxCommitArrayRange(array, N/2, N, p) );
    }
    /* Check */
    for (i = 0; i < N/2; i++) {
        ASSERT(localArray2[3*i].x == 0);
        ASSERT(localArray2[3*i].y == 0);
        ASSERT(localArray2[3*i+1].x == 0);
        ASSERT(localArray2[3*i+1].y == 0);
        ASSERT(localArray2[3*i+2].x == 0);
        ASSERT(localArray2[3*i+2].y == 0);
    }
    for (i = N/2; i < N; i++) {
        ASSERT(localArray2[3*i].x == i);
        ASSERT(localArray2[3*i].y == i);

       /* Unchanged in between */
        ASSERT(localArray2[3*i+1].x == 0);
        ASSERT(localArray2[3*i+1].y == 0);
        ASSERT(localArray2[3*i+2].x == 0);
        ASSERT(localArray2[3*i+2].y == 0);
    }

    VX_CALL( vxReleaseArray(&array) );
    ASSERT( array == 0);
}

TESTCASE_TESTS(Array,
               testAccessCopyWrite,
               testAccessCopyRead
               )


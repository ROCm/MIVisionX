/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
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

TESTCASE(SmokeTest, CT_VXContext, ct_setup_vx_context, 0)

typedef struct _mystruct {
    vx_uint32 some_uint;
    vx_float64 some_double;
} mystruct;

TEST(SmokeTest, test_vxRegisterUserStruct)
{
    vx_context context = context_->vx_context_;
    vx_enum mytype = 0;
    vx_array array = 0;
    vx_enum type = 0;
    vx_size sz = 0;

    mytype = vxRegisterUserStruct(context, sizeof(mystruct));
    ASSERT(mytype >= VX_TYPE_USER_STRUCT_START);

    ASSERT_VX_OBJECT(array = vxCreateArray(context, mytype, 10), VX_TYPE_ARRAY);

    VX_CALL(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)));
    ASSERT_EQ_INT(mytype, type);

    VX_CALL(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &sz, sizeof(sz)));
    ASSERT_EQ_INT(sizeof(mystruct), sz);

    VX_CALL(vxReleaseArray(&array));
    ASSERT(array == 0);
}


TEST(SmokeTest, test_vxHint)
{
    vx_image image = 0;
    vx_graph graph = 0;
    vx_context context = vxCreateContext(); // don't use global context to avoid side effects on other tests

    ASSERT_VX_OBJECT(context, VX_TYPE_CONTEXT);
    CT_RegisterForGarbageCollection(context, ct_destroy_vx_context, CT_GC_OBJECT);

    ASSERT_VX_OBJECT(image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    vxHint((vx_reference)image, VX_HINT_SERIALIZE);
    vxHint((vx_reference)graph, VX_HINT_SERIALIZE);
    vxHint((vx_reference)context, VX_HINT_SERIALIZE);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_EQ_PTR(0, image);
    ASSERT_EQ_PTR(0, graph);
}


TESTCASE_TESTS(SmokeTest,
        test_vxRegisterUserStruct,
        test_vxHint
        )

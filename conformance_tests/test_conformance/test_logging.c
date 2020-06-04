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

TESTCASE(Logging, CT_VoidContext, 0, 0)

static vx_bool log_callback_is_called = vx_false_e;
static void VX_CALLBACK test_log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    log_callback_is_called = vx_true_e;
    ASSERT_(printf("\tActual: %d < %d\n\n", (int)strlen(string), (int)VX_MAX_LOG_MESSAGE_LEN), strlen(string) < VX_MAX_LOG_MESSAGE_LEN);
}

TEST(Logging, Cummulative)
{
    vx_image image = 0;
    vx_context context = vxCreateContext();

    ASSERT_VX_OBJECT(context, VX_TYPE_CONTEXT);
    CT_RegisterForGarbageCollection(context, ct_destroy_vx_context, CT_GC_OBJECT);
    ASSERT_VX_OBJECT(image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    // normall logging
    vxRegisterLogCallback(context, test_log_callback, vx_false_e);
    log_callback_is_called = vx_false_e;
    vxAddLogEntry((vx_reference)image, VX_FAILURE, "hello world", 1, 2, 3);
    ASSERT(log_callback_is_called);

    // clear callback
    vxRegisterLogCallback(context, NULL, vx_true_e);
    log_callback_is_called = vx_false_e;
    vxAddLogEntry((vx_reference)image, VX_FAILURE, "hello world", 4, 5, 6);
    ASSERT(!log_callback_is_called);

    // restore callback
    vxRegisterLogCallback(context, test_log_callback, vx_true_e);

    // disable logs for image
    VX_CALL(vxDirective((vx_reference)image, VX_DIRECTIVE_DISABLE_LOGGING));
    log_callback_is_called = vx_false_e;
    vxAddLogEntry((vx_reference)image, VX_FAILURE, "hello world", 4, 5, 6);
    ASSERT(!log_callback_is_called);

    // turn on logs once again
    VX_CALL(vxDirective((vx_reference)image, VX_DIRECTIVE_ENABLE_LOGGING));
    log_callback_is_called = vx_false_e;
    vxAddLogEntry((vx_reference)image, VX_FAILURE, "%*s", VX_MAX_LOG_MESSAGE_LEN + 20, ""); // 20 symbols longer string than limit
    ASSERT(log_callback_is_called);

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);
}


TESTCASE_TESTS(Logging,
        Cummulative
        )

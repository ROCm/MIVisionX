# Copyright (c) 2014 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

CTS_VERSION_FILE := $(strip $(wildcard $(LOCAL_PATH)/../../openvx_cts_version.inc))
ifdef CTS_VERSION_FILE
    LOCAL_CFLAGS += -DHAVE_VERSION_INC
endif

FILE_LIST := $(wildcard $(LOCAL_PATH)/../../test_engine/*.c) $(wildcard $(LOCAL_PATH)/../../test_conformance/*.c)
LOCAL_SRC_FILES := $(FILE_LIST:$(LOCAL_PATH)/%=%)
LOCAL_C_INCLUDES := $(OPENVX_INCLUDES) $(LOCAL_PATH)/../../ $(LOCAL_PATH)/../../test_conformance
LOCAL_LDLIBS := $(OPENVX_LIBRARIES)
LOCAL_MODULE := vx_test_conformance
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_CFLAGS += -DHAVE_NEON=1 -march=armv7-a -mfpu=neon -ftree-vectorize -ffast-math -mfloat-abi=softfp
endif
ifneq ($(CT_DISABLE_TIME_SUPPORT),1)
    LOCAL_CFLAGS += -DCT_TEST_TIME
endif
$(info ${LOCAL_CFLAGS})
include $(BUILD_EXECUTABLE)

/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

 /* Steps:
 *
 *  1. create engin and stream
 *  2. create user memory (source, weights, destination)
 *  3. create memory descriptor
 *  4. create convolution descriptor
 *  5. create convolution primitive descriptor
 *  6. create convolution primitive
 *  7. execute the convlution
 *  8. create ReLU desciptor
 *  9. create ReLU primitive descriptor
 *  10. create ReLU primitive
 *  11. execute ReLU
 */

//IMP => export ZENDNN_VERBOSE=1
//ZENDNN_VERBOSE=1 bin/simple_conv_test cpu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <math.h>
#include <cstdlib>
#include <unistd.h>
#include <string.h>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define ZENDNN_CMP_OUTPUT   1

using namespace std;
using namespace zendnn;


//set stride and padding
const memory::dims strides = {2, 2};
const memory::dims padding = {2, 2};


float *transpose(const float *matrix, int n, int m, float *transposed) {
    int i = 0;
    int j = 0;
    float num;
    //float *transposed=(float *)malloc(sizeof(float)*n*m);
    if (transposed == NULL) {
        zendnnError(ZENDNN_ALGOLOG, "transpose Memory Error");
    }
    while (i < n) {
        j = 0;
        while (j < m) {
            num = *(matrix + i*m + j);
            *(transposed + i+n*j) = num;
            j++;
        }
        i++;
    }

    return transposed;
}

void nchw2nhwc(zendnn::memory &in,int N, int C, int H, int W, zendnn::memory &out) {
    float *in_data= (float *)in.get_data_handle();
    float *out_data= (float *)out.get_data_handle();


    for (int n = 0; n < N; n++) {
        int in_batch_offset = n * C * H * W;
        int out_batch_offset = n * H * W * C;
        for (int c = 0; c < C; ++c) {
            int in_ch_offset = c * H * W + in_batch_offset;
            int out_ch_offset = out_batch_offset + c;
            for (int h = 0; h < H; ++h) {
                int in_row_offset = h * W + in_ch_offset;
                int out_row_offset = out_ch_offset + h * W * C;
                for (int w = 0; w < W; ++w) {
                    int in_addr = w + in_row_offset;
                    int out_addr = out_row_offset + w * C;
                    out_data[out_addr] = in_data[in_addr];
                }
            }
        }
    }
}


//function to init data
void init_data(memory &m, float v) {
    size_t size = m.get_desc().get_size() /sizeof(float);
    //std::vector<float> data(size);
    srand(1111);
    float *data= (float *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        data[i] = rand()%5;
    }
}

//function to exeute non-fused relu
void create_and_execute_relu(memory &data, engine &eng, stream &s) {
    //relu operates on whatever data format is given to it

    //create a desciptor
    auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                                        algorithm::eltwise_relu, data.get_desc(), 0.f, 0.f);
    //create a relu primitive descriptor
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    //create a relu primitive
    auto relu = eltwise_forward(relu_pd);

    //execute in-place
    relu.execute(s, {{ZENDNN_ARG_SRC, data}, {ZENDNN_ARG_DST, data}});
}

// Implementation for the naive convlolution on the nchw (data) and oihw (weights),
// followed by execution of non-fused relu
void conv_relu_naive(memory user_src, memory user_wei, memory user_dst, memory user_bias,
                     engine &eng, stream &s) {
    //Create mem_desc
    //copy the dimesnions and formats from user's memory
    //First it sets the dimensions and format for the convolution memory descriptor (_md) to match user_ value
    //for source, destonation and weight data.
    //Then it uses those md to create the convolution descriptor conv_d which tells ZENDNN to use plain foramt
    //(NCHW) for the convolution
    auto conv_src_md = memory::desc(user_src.get_desc());
    auto conv_wei_md = memory::desc(user_wei.get_desc());
    auto conv_dst_md = memory::desc(user_dst.get_desc());
    auto conv_bias_md = memory::desc(user_bias.get_desc());

    //Next program creates a convolution descriptor, primitive descriptor conv_pd and convolution primitive conv
    //These structs will inherit NCHW format from md by way of conv_d
    //Finally it creates the convolution primitive conv  and adds it to stream s, and then executes the
    //create_and_execute_relu(user_dst) pfunction

    //create a convolution descriptor
    auto conv_d = convolution_forward::desc(prop_kind::forward_inference,
#if !ZENDNN_ENABLE
                                            algorithm::convolution_direct,
#else
                                            algorithm::convolution_gemm,
#endif
                                            conv_src_md, conv_wei_md, conv_bias_md,
                                            conv_dst_md, strides, padding, padding);

    //create a convolution primitive descriptor
    auto conv_pd = convolution_forward::primitive_desc(conv_d, eng);

    //create convolution primitive
    auto conv = convolution_forward(conv_pd);

    //execute convolution by adding it to the stream s
    conv.execute(s, {
        {ZENDNN_ARG_SRC, user_src}, {ZENDNN_ARG_WEIGHTS, user_wei},{ZENDNN_ARG_BIAS, user_bias},
        {ZENDNN_ARG_DST, user_dst}
    });

    //execute relu (on convolution's destination format, whatever it is )
    //create_and_execute_relu(user_dst, eng, s);
}


int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test starts");
    ofstream file;

    //initialize the engine
    if (argc <= 1) {
        zendnnInfo(ZENDNN_TESTLOG, "The command to run this test");
        zendnnInfo(ZENDNN_TESTLOG, "ZENDNN_VERBOSE=1 bin/simple_conv_test cpu");

        return -1;
    }

    zendnn::engine::kind engine_kind = parse_engine_kind(argc, argv, 1);

    //initialize engine
    zendnn::engine eng(engine_kind, 0);
    zendnnInfo(ZENDNN_TESTLOG, "engine created");

    //initialize stream
    stream s(eng);
    zendnnInfo(ZENDNN_TESTLOG, "stream created");


    //Set dimensions for synthetic data and weights
    const memory::dim BATCH = 100;
    const memory::dim IC = 3, OC = 64;
    const memory::dim IH = 224, KH = 7, OH = 111;
    const memory::dim IW = 224, KW = 7, OW = 111;


    int batch = 100;
    int input_h = 224;
    int kernel_h = 7;
    int output_h = 111;
    int padding = 2;
    int stride = 2;
    int channel = 3;
    int filters = 64;

    //const memory::dim IH = 224, KH = 1, OH = 114;
    //const memory::dim IW = 224, KW = 1, OW = 114;
    //create ZENDNN memory objects in NCHW format. They are called user_ because they are meant to represent
    //the user's source data entering ZENDNN with NCHW format
    //create ZENDNN memory objct for user's tensors (in nchw and oihw format)
    //here library allocate memory
    auto user_src = memory({{BATCH, IC, IH, IW}, memory::data_type::f32,
        memory::format_tag::nhwc
    },
    eng);

#if !ZENDNN_ENABLE
    auto user_wei = memory({{OC, IC, KH, KW}, memory::data_type::f32,
        memory::format_tag::oihw
    },
    eng);
#else
    auto user_wei3 = memory({{OC, IC, KH, KW}, memory::data_type::f32,
        memory::format_tag::oihw
    },
    eng);
    auto user_wei2 = memory({{OC, IC, KH, KW}, memory::data_type::f32,
        memory::format_tag::oihw
    },
    eng);
    auto user_wei = memory({{OC, IC, KH, KW}, memory::data_type::f32,
        memory::format_tag::oihw
    },
    eng);
#endif

    auto user_dst = memory({{BATCH, OC, OH, OW}, memory::data_type::f32,
        memory::format_tag::nhwc
    },
    eng);
    auto user_bias = memory({{OC}, memory::data_type::f32,
        memory::format_tag::x
    }, eng);

    zendnnInfo(ZENDNN_TESTLOG, "ZENDNN memory objects created");
    //Fill source, destination and weights with synthetic data
    init_data(user_src, 1);
    init_data(user_dst, -1);

#if !ZENDNN_ENABLE
    init_data(user_wei, .5);
#else
    init_data(user_wei3, .5);
    nchw2nhwc(user_wei3, OC, IC, KH, KW, user_wei2);
    float *in_data= (float *)user_wei2.get_data_handle();
    transpose(in_data, OC, IC*KH*KW, (float *)user_wei.get_data_handle());
#endif

    init_data(user_bias, .5);

    zendnnInfo(ZENDNN_TESTLOG, "implementation: naive");
    //run conv + relu without fusing
    conv_relu_naive(user_src, user_wei, user_dst, user_bias, eng, s);
    zendnnInfo(ZENDNN_TESTLOG, "conv + relu w/ nchw completed");

    //Dump the output buffer to a file
    const char *zenDnnRootPath = getenv("ZENDNN_GIT_ROOT");
#if !ZENDNN_ENABLE
    file.open(zenDnnRootPath + std::string("/_out/tests/ref_conv_output"));
#else
    file.open(zenDnnRootPath + std::string("/_out/tests/zendnn_conv_output"));
#endif //ZENDNN_ENABLE

    double sum = 0;
    size_t size = user_dst.get_desc().get_size() /sizeof(float);
    float *dataHandle= (float *)user_dst.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        sum += dataHandle[i];
    }

    //write the dataHandle to the file
    file.write(reinterpret_cast<char const *>(dataHandle), user_dst.get_desc().get_size());
    file.close();

#if !ZENDNN_ENABLE
    zendnnInfo(ZENDNN_TESTLOG, "Ref API test: SUM: ", sum);
    string str = std::string("sha1sum ") + zenDnnRootPath +
        std::string("/_out/tests/ref_conv_output > ") + zenDnnRootPath +
        std::string("/tests/api_tests/sha_out_NHWC/ref_conv_output.sha1");
#else
    zendnnInfo(ZENDNN_TESTLOG, "ZENDNN API test: SUM: ", sum);
    string str = std::string("sha1sum ") + zenDnnRootPath +
        std::string("/_out/tests/zendnn_conv_output > ") + zenDnnRootPath +
        std::string("/_out/tests/zendnn_conv_output.sha1");
#endif

    //Convert string to const char * as system requires
    //parameters of the type const char *
    const char *command = str.c_str();
    int status = system(command);

#if ZENDNN_CMP_OUTPUT  //compare SHA1 value
    ifstream zenDnnSha1(zenDnnRootPath + std::string("/_out/tests/zendnn_conv_output.sha1"));
    string firstWordZen;

    while (zenDnnSha1 >> firstWordZen) {
        zendnnInfo(ZENDNN_TESTLOG, "ZenDNN output SHA1 value: ", firstWordZen);
        zenDnnSha1.ignore(numeric_limits<streamsize>::max(), '\n');
    }


    ifstream refSha1(zenDnnRootPath +
            std::string("/tests/api_tests/sha_out_NHWC/ref_conv_output.sha1"));
    string firstWordRef;

    while (refSha1 >> firstWordRef) {
        zendnnInfo(ZENDNN_TESTLOG, "Ref output SHA1 value: ", firstWordRef);
        refSha1.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    ZENDNN_CHECK(firstWordZen == firstWordRef, ZENDNN_TESTLOG,
                 "sha1 /sha1sum value of ZenDNN output and Ref output do not matches");
#endif //ZENDNN_CMP_OUTPUT compare SHA1 value

    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test ends");
    return 0;
}

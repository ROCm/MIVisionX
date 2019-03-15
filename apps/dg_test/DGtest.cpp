#include "DGtest.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <unistd.h>
#if ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); exit(-1); } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

DGtest::DGtest(const char* weights, std::string inputFile, std::string outputFile, const int batchSize) : mWeights(weights), mBatchSize(batchSize){
    mContext = vxCreateContext();
    if(vxGetStatus((vx_reference)mContext)) {
        std::cerr << "ERROR: vxCreateContext(...) failed" << std::endl;
        exit(-1);
    }
    mInputTensor = std::make_unique<VXtensor>(mContext, mBatchSize, inputFile, VX_READ_ONLY);
    
    mGraph = vxCreateGraph(mContext);
    if(vxGetStatus((vx_reference)mGraph)) {
        std::cerr << "ERROR: vxCreateGraph(...) failed" << std::endl;
        exit(-1);
    }
    mOutputTensor = std::make_unique<VXtensor>(mContext, mBatchSize, outputFile, VX_WRITE_ONLY);
};

DGtest::~DGtest(){

    ERROR_CHECK_STATUS(vxReleaseContext(&mContext));
    printf("DGtest successful\n");
};


void DGtest::runInference() {
    //read in from the specified input tensor
    if(mInputTensor->readTensor() < 0) {
        std::cout << "Failed to initialize tensor data from " << mInputTensor->getFileName() << std::endl;
        exit(-1);
    }
    std::cout << "OK: initialized tensor 'data' from " << mInputTensor->getFileName() << std::endl;
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();
    vxRegisterLogCallback(mContext, log_callback, vx_false_e);
    
    //add input tensor, output tensor, and weights to the graph
    vx_status status = annAddToGraph(mGraph, mInputTensor->getTensor(), mOutputTensor->getTensor(), mWeights);
    if(status) {
         printf("ERROR: annAddToGraph() failed (%d)\n", status);
         exit(-1);
    }

    //verify the graph
    status = vxVerifyGraph(mGraph);
    if(status) {
        printf("ERROR: vxVerifyGraph(...) failed (%d)\n", status);
        exit(-1);
    }
    t1 = clockCounter();
    printf("OK: graph initialization with annAddToGraph() took %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    //process the graph
    t0 = clockCounter();
    status = vxProcessGraph(mGraph);
    t1 = clockCounter();
    if(status != VX_SUCCESS) {
        printf("ERROR: vxProcessGraph() failed (%d)\n", status);
        exit(-1);
    }
    printf("OK: vxProcessGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);
    
    //write out to the specified output tensor
    if(mOutputTensor->writeTensor() < 0) {
        std::cout << "Failed to write tensor data to " << mOutputTensor->getFileName() << std::endl;
        exit(-1);
    }
    std::cout << "OK: wrote tensor 'loss' into " << mOutputTensor->getFileName() << std::endl;

    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
        status = vxProcessGraph(mGraph);
        if(status != VX_SUCCESS)
            break;
    }
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);
    ERROR_CHECK_STATUS(vxReleaseGraph(&mGraph));
}
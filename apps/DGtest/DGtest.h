#pragma once

#include "annmodule.h"
#include "VXtensor.h"
#include <memory>

/**
 *  Class to run the inference
 */

class DGtest
{
public:
    DGtest(const char* weights, std::string inputFile, std::string outputFile, const int batchSize);
    ~DGtest();

    /**
     *  Run the inference
     */
    void runInference(); 

private:

    /**
     *  Weights file name
     */
    const char* mWeights;

    /**
     *  The pointer to the input tensor object
     */
    std::unique_ptr<VXtensor> mInputTensor;
    
    /**
     *  The pointer to the output tensor object
     */
    std::unique_ptr<VXtensor> mOutputTensor;

    /**
     *  The batch size to run the inference
     */
    int mBatchSize;

    /**
     *  Context that will be used for the inference
     */
    vx_context mContext;
    
    /**
     *  Graph that will be used for the inference
     */
    vx_graph mGraph;
};
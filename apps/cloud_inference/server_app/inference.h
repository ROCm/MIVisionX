#ifndef INFERENCE_H
#define INFERENCE_H

#include "arguments.h"
#include "infcom.h"
#include "profiler.h"
#include "region.h"
#include <string>
#include <tuple>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <VX/vx.h>
#include <vx_ext_amd.h>

// inference scheduler modes
//   NO_INFERENCE_SCHEDULER    - no scheduler (i.e., network connection with respond back immediately)
//   LIBRE_INFERENCE_SCHEDULER - simple free flow scheduler that makes use several messaging queues and threads
#define NO_INFERENCE_SCHEDULER        0
#define LIBRE_INFERENCE_SCHEDULER     1

// configuration
//   INFERENCE_SCHEDULER_MODE     - pick one of the modes from above
//   INFERENCE_SERVICE_IDLE_TIME  - inference service idle time (milliseconds) if there is no activity
#define INFERENCE_SCHEDULER_MODE       LIBRE_INFERENCE_SCHEDULER
#define INFERENCE_SERVICE_IDLE_TIME    1
#define DEVICE_QUEUE_FULL_SLEEP_MSEC   1  // msec to sleep when device queue is full

// inference scheduler configuration
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
#define DONOT_RUN_INFERENCE            0  // for debugging
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
#define INFERENCE_PIPE_QUEUE_DEPTH     5  // inference pipe queue depth
#define MAX_INPUT_QUEUE_DEPTH       1024  // max number of images in input Q
#define MAX_DEVICE_QUEUE_DEPTH      1024  // max number of images in device Q
#define USE_SSE_OPTIMIZATION           1  // enable/disable SSE intrinsics for resize and format conversion
#define DONOT_RUN_INFERENCE            0  // for debugging
#define USE_ADVANCED_MESSAGE_Q         0  // experimental code
#endif

// Bounding box region:: todo add this as parameters to app and pass it to server
#define BOUNDING_BOX_CONFIDENCE_THRESHHOLD  0.2
#define BOUNDING_BOX_NMS_THRESHHOLD         0.4
#define BOUNDING_BOX_NUMBER_OF_CLASSES      20


extern "C" {
    typedef VX_API_ENTRY vx_graph VX_API_CALL type_annCreateGraph(
            vx_context context,
            vx_tensor input,
            vx_tensor output,
            const char * options
        );
};

extern "C" {
    typedef VX_API_ENTRY vx_status VX_API_CALL type_annAddToGraph(vx_graph graph, vx_tensor input, vx_tensor output, const char * binaryFilename);
};

template<typename T>
class MessageQueue {
public:
    MessageQueue() : enqueueCount{ 0 }, dequeueCount{ 0 }, maxQueueDepth{ 0 } {
    }
    void setMaxQueueDepth(int maxDepth) {
        maxQueueDepth = maxDepth;
    }

    size_t size() {
        return enqueueCount - dequeueCount;
    }
    void enqueue(T const& value) {
        if(maxQueueDepth > 0) {
            // make sure that device queue are stay within the limit
            while(size() >= maxQueueDepth) {
                std::this_thread::sleep_for(std::chrono::milliseconds(DEVICE_QUEUE_FULL_SLEEP_MSEC));
            }
        }
        mutex.lock();
        queue.push(value);
        enqueueCount++;
        mutex.unlock();
        signal.notify_one();
    }
    void dequeue(T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        while(queue.empty()) {
            signal.wait(lock);
        }
        value = queue.front();
        queue.pop();
        dequeueCount++;
    }

private:
    int enqueueCount;
    int dequeueCount;
    int maxQueueDepth;
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable signal;
};

#if USE_ADVANCED_MESSAGE_Q
template<typename T>
class MessageQueueAdvanced {
public:
    MessageQueueAdvanced(int maxSize) : count{ 0 }, maxQSize(maxSize), end_of_sequence(0){
    }
    size_t size() {
        return count;
    }
    void enqueue(T const& value) {
        while (true){
            if (size() < maxQSize){
                q_mtx.lock();
                queue.push(value);
                count++;
                q_mtx.unlock();
                signal.notify_one();
                break;
            }else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(DEVICE_QUEUE_FULL_SLEEP_MSEC));
            }
        }
    }
    void dequeue(T& value) {
        std::unique_lock<std::mutex> lock(q_mtx);
        while (count <= 0) {
            signal.wait(lock);
        }
        value = queue.front();
        queue.pop();
        count--;
    }
    void dequeueBatch(int batchsize, std::vector<T>& BatchQ){
        while (true){
            std::unique_lock<std::mutex> lock(q_mtx);
            //pop batch
            if (count >= batchsize)
            {
                for (int i = 0; i < batchsize; i++){
                    BatchQ.push_back(queue.front());
                    queue.pop();
                    count--;
                }
                break;
            }
            else if (end_of_sequence)
            {
                // pop remaining
                int size_rem = count;
                for (int i = 0; i < size_rem; i++){
                    BatchQ.push_back(queue.front());
                    queue.pop();
                    count--;
                }
                break;
            }
            else
            {
                signal.wait(lock);
            }
        }
    }
    void endOfSequence(){
        std::lock_guard<std::mutex> lock(q_mtx);
        end_of_sequence = 1;
    }
private:
    int count;
    int maxQSize;
    int end_of_sequence;
    std::queue<T> queue;
    std::mutex q_mtx;
    std::condition_variable signal;
};
#endif

class InferenceEngine {
public:
    InferenceEngine(int sock, Arguments * args, std::string clientName, InfComCommand * cmd);
    ~InferenceEngine();
    int run();

protected:
    // scheduler thread workers
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
    // no separate threads needed
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    // libre scheduler needs:
    //   masterInputQ thread
    //   device threads for input copy, processing, and output copy
    void workMasterInputQ();
    void workDeviceInputCopy(int gpu);
    void workDeviceProcess(int gpu);
    void workDeviceOutputCopy(int gpu);
#endif

private:
    void dumpBuffer(cl_command_queue cmdq, cl_mem mem, std::string fileName);

private:
    // configuration
    int sock;
    Arguments * args;
    std::string modelName;
    std::string options;
    int GPUs;
    int dimInput[3];
    int dimOutput[3];
    bool useShadowFilenames;
    bool receiveFileNames;
    int topK;
    int reverseInputChannelOrder;
    float preprocessMpy[3];
    float preprocessAdd[3];
    std::string clientName;
    std::string modelPath;
    std::string modulePath;
    void * moduleHandle;
    type_annCreateGraph * annCreateGraph;
    type_annAddToGraph  * annAddtoGraph;
    cl_device_id device_id[MAX_NUM_GPU];
    int batchSize;
    int inputSizeInBytes;
    int outputSizeInBytes;
    bool deviceLockSuccess;
    int detectBoundingBoxes;
    int useFp16, numDecThreads;
    CYoloRegion *region;
    // scheduler output queue
    //   outputQ: output from the scheduler <tag,label>
    MessageQueue<std::tuple<int,int>>     outputQ;
    MessageQueue<std::vector<unsigned int>>        outputQTopk;      // outputQ for topK vec<tag, top_k labels>
    MessageQueue<std::vector<ObjectBB>> OutputQBB;

    vx_status DecodeScaleAndConvertToTensor(vx_size width, vx_size height, int size, unsigned char *inp, float *out, int use_fp16=0);
    void DecodeScaleAndConvertToTensorBatch(std::vector<std::tuple<char*, int>>& batch_Q, int start, int end, int dim[3], float *tens_buf);
    void RGB_resize(unsigned char *Rgb_in, unsigned char *Rgb_out, unsigned int swidth, unsigned int sheight, unsigned int sstride, unsigned int dwidth, unsigned int dheight);

#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER && !DONOT_RUN_INFERENCE
    // OpenVX resources
    vx_context openvx_context;
    vx_tensor openvx_input;
    vx_tensor openvx_output;
    vx_graph openvx_graph;
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    // master scheduler thread
    std::thread * threadMasterInputQ;
    // scheduler thread objects
    std::thread * threadDeviceInputCopy[MAX_NUM_GPU];
    std::thread * threadDeviceProcess[MAX_NUM_GPU];
    std::thread * threadDeviceOutputCopy[MAX_NUM_GPU];
    //   inputQ: input to the scheduler <tag,byteStream,size>
#if  USE_ADVANCED_MESSAGE_Q
    MessageQueueAdvanced<std::tuple<int,char *,int>> inputQ;
    // scheduler device queues
    MessageQueueAdvanced<int>                    * queueDeviceTagQ[MAX_NUM_GPU];
    MessageQueueAdvanced<std::tuple<char *,int>> * queueDeviceImageQ[MAX_NUM_GPU];
#else
    MessageQueue<std::tuple<int,char *,int>> inputQ;
    // scheduler device queues
    MessageQueue<int>                    * queueDeviceTagQ[MAX_NUM_GPU];
    MessageQueue<std::tuple<char *,int>> * queueDeviceImageQ[MAX_NUM_GPU];
#endif
    MessageQueue<cl_mem>                 * queueDeviceInputMemIdle[MAX_NUM_GPU];
    MessageQueue<cl_mem>                 * queueDeviceInputMemBusy[MAX_NUM_GPU];
    MessageQueue<cl_mem>                 * queueDeviceOutputMemIdle[MAX_NUM_GPU];
    MessageQueue<cl_mem>                 * queueDeviceOutputMemBusy[MAX_NUM_GPU];
    // scheduler resources
    cl_context opencl_context[MAX_NUM_GPU];
    cl_command_queue opencl_cmdq[MAX_NUM_GPU];
    vx_context openvx_context[MAX_NUM_GPU];
    vx_graph openvx_graph[MAX_NUM_GPU];
    vx_tensor openvx_input[MAX_NUM_GPU];
    vx_tensor openvx_output[MAX_NUM_GPU];
#endif
};

#endif

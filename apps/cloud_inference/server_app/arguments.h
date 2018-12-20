#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include "common.h"
#include <vector>
#include <string>
#include <tuple>
#include <mutex>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cl.h>

#define MAX_DEVICE_USE_LIMIT     1   // number of parallel sessions allowed per device

class Arguments {
public:
    Arguments();
    ~Arguments();
    int initializeConfig(int argc, char * argv[]);

    // configuration
    int getPort() {
        return port;
    }
    int getBatchSize() {
        return batchSize;
    }
    int getNumGPUs() {
        return numGPUs;
    }
    const std::string& getConfigurationDir() {
        return configurationDir;
    }
    cl_platform_id getPlatformId() {
        return platform_id;
    }
    const std::string& getlocalShadowRootDir() {
        return localShadowRootDir;
    }

    const std::string& getModelCompilerPath() {
        return modelCompilerPath;
    }

    // global mutex
    void lock() {
        mutex.lock();
    }
    void unlock() {
        mutex.unlock();
    }

    // model configurations
    int getNumConfigureddModels() {
        std::lock_guard<std::mutex> lock(mutex);
        return configuredModels.size();
    }
    int getNumUploadedModels() {
        std::lock_guard<std::mutex> lock(mutex);
        return uploadedModels.size();
    }
    std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string> getConfiguredModelInfo(int index) {
        std::lock_guard<std::mutex> lock(mutex);
        return configuredModels[index];
    }
    std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float> getUploadedModelInfo(int index) {
        std::lock_guard<std::mutex> lock(mutex);
        return uploadedModels[index];
    }
    int getNextModelUploadCounter() {
        std::lock_guard<std::mutex> lock(mutex);
        modelFileDownloadCounter++;
        saveConfig();
        return modelFileDownloadCounter;
    }
    void addConfigToUploadedList(std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float>& ann) {
        std::lock_guard<std::mutex> lock(mutex);
        uploadedModels.push_back(ann);
    }
    void addConfigToPreconfiguredList(std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>& ann) {
        std::lock_guard<std::mutex> lock(mutex);
        bool replaced = false;
        for(auto it = configuredModels.begin(); it != configuredModels.end(); it++) {
            if(std::get<0>(*it) == std::get<0>(ann)) {
                *it = ann;
                replaced = true;
                break;
            }
        }
        if(!replaced) {
            configuredModels.push_back(ann);
        }
    }
    bool checkPassword(std::string code) {
        return (password == code) ? true : false;
    }
    // set localShadowRootDir (full absolute path)
    void setLocalShadowRootDir(const std::string& localShadowDir)
    {
        localShadowRootDir = localShadowDir;
    }
    // set modelCompiler(nnir) path
    void setModelCompilerPath(const std::string& modelCompDir)
    {
        modelCompilerPath = modelCompDir;
    }
    
    bool fp16Inference()
    {
        return useFp16Inference;
    }
    int decThreads()
    {
        return numDecThreads;
    }

    // device resources
    int lockGpuDevices(int GPUs, cl_device_id * device_id_);
    void releaseGpuDevices(int GPUs, const cl_device_id * device_id_);

protected:
    void setConfigurationDir();
    void loadConfig();
    void saveConfig();
    void getPreConfiguredModels();

private:
    // loaded configuration
    std::string workFolder;
    int modelFileDownloadCounter;
    int port;
    int batchSize;
    int maxPendingBatches;
    int numGPUs;
    int useFp16Inference;
    int numDecThreads;
    int gpuIdList[MAX_NUM_GPU];
    std::string password;
    // derived configuration
    int maxGpuId;
    cl_uint num_devices;
    int deviceUseCount[MAX_NUM_GPU];
    cl_device_id device_id[MAX_NUM_GPU];
    cl_platform_id platform_id;
    std::string configurationFile;
    std::string configurationDir;
    std::string localShadowRootDir;
    std::string modelCompilerPath;
    std::vector<std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>> configuredModels;
    std::vector<std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float>> uploadedModels;
    // misc
    std::mutex mutex;
};

#endif

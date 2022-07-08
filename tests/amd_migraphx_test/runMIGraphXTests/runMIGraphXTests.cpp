#include "vx_amd_migraphx.h"
#include <cstring>
#include <random>
#include <fstream>
#include <algorithm>
#include <string>
#define MAX_STRING_LENGTH 100

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#endif

#define ERROR_CHECK_STATUS(status) { \
    vx_status status_ = (status); \
    if (status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

#define ERROR_CHECK_OBJECT(obj) { \
    vx_status status_ = vxGetStatus((vx_reference)(obj)); \
    if(status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strnlen(string, MAX_STRING_LENGTH);
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

void show_usage() {
    printf(
            "\n"
            "Usage: ./runmigraphxtests \n"
            "--imagenet_image          <path to image from imagenet dataset>\n"
            "--mnist_image             <path to image from mnist dataset>\n"
            "--profiler_mode  <range:0-2; default:0> [optional]\n"
            "  Mode 0 - Run all tests\n"
            "  Mode 1 - Run all ONNX tests\n"
            "  Mode 2 - Run all JSON tests\n"
            "--profiler_level <range:0-N; default:1> [N = batch size][optional]\n"
            "--mnist          <mnist-model>     \n"
            "--resnet50       <resnet50-model>  \n"
            "--googlenet      <googlenet-model> \n" 
            "--squeezenet     <resnet101-model> \n"
            "--alexnet        <resnet152-model> \n"
            "--vgg19          <vgg19-model>     \n"
            "--densenet       <densenet-model>  \n"
            "\n"
        ); 
}

int main(int argc, char **argv) {

    // check command-line usage
    std::string binaryFilename_squeezenet_str;
    std::string binaryFilename_resnet50_str;
    std::string binaryFilename_vgg19_str;
    std::string binaryFilename_googlenet_str;
    std::string binaryFilename_alexnet_str;
    std::string binaryFilename_densenet_str;
    std::string binaryFilename_mnist_str;
    std::string imagenet_inputFile_str;
    std::string mnist_inputFile_str;

    int parameter = 0;
    int64_t freq = clockFrequency(), t0, t1;
    int N = 1000;
    bool runMnist = false, runResnet50 = false, runVgg19 = false, runGooglenet = false, runDensenet = false, runAlexnet = false, runSqueezenet = false, runAnyImagenet = false;

    for(int arg = 1; arg < argc; arg++) {
        if (!strcasecmp(argv[arg], "--help") || !strcasecmp(argv[arg], "--H") || !strcasecmp(argv[arg], "--h")) {
            show_usage();
            exit(-1);
        }
        else if (!strcasecmp(argv[arg], "--imagenet_image")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing image for imagenet model's file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            arg++;
            imagenet_inputFile_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--mnist_image")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing image for mnist model's file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            arg++;
            mnist_inputFile_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--profiler_mode")) {
            int profiler_mode = 0;
            arg++;
            profiler_mode = std::stoi(argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--profiler_level")){
            int profiler_level = 1;
            arg++;
            profiler_level = std::stoi(argv[arg]);
            if(1 < profiler_level <= 10) {
                printf("\n\nERROR: profiler level has to be between 1-10)\n\n\n");
                exit(-1);
            }
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--mnist")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing mnist ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runMnist = true;
            arg++;
            binaryFilename_mnist_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--alexnet")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing alexnet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runAlexnet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_alexnet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--squeezenet")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing squeezenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runSqueezenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_squeezenet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--resnet50")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing resnet50 ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runResnet50 = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_resnet50_str = (argv[arg]);
            parameter++;
        }

        else if (!strcasecmp(argv[arg], "--vgg19")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing vgg19 ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runVgg19 = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_vgg19_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--googlenet")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing googlenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runGooglenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_googlenet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--densenet")) {
            if ((arg + 1) == argc) {
                printf("\n\nERROR: missing densenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runDensenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_densenet_str = (argv[arg]);
            parameter++;
        }
    }

    if (parameter < 2) {
        printf("\nERROR: missing parameters in command-line.\n");
        show_usage();
        exit(-1);
    }

    // create context, input, output, and graph
    vx_status status = 0;
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    
    // load vx_nn kernels
    ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_amd_migraphx"));

    // initialize variables
    vx_tensor input_tensor_224x224, input_tensor_28x28;
    vx_size input_num_of_dims = 4;
    vx_size input_dims_data_224x224[4] = {128, 3, 224, 224};
    vx_size input_dims_data_28x28[4] = {28, 28, 1, 1}; 
    vx_size output_num_of_dims_2 = 2;
    vx_size output_num_of_dims_4 = 4;
    vx_size output_dims_data_1x1000[2] = {128, 1000};
    vx_size output_dims_data_1x1000x1x1[4] = {128, 1000, 1, 1};
    vx_size output_dims_data_1x10[2] = {1, 10};
    vx_size stride[4];
    vx_map_id map_id;
    void *ptr = nullptr;
    auto num_results_imagenet = 1000;

    //imagenet label file
    std::string labelText[1000];
    std::string imagenetLabelFileName = ("../labels.txt");

    std::string line;
    std::ifstream out(imagenetLabelFileName);
    if(!out) {
      std::cout << "label file failed to open" << std::endl;
      return -1; 
    }
    int lineNum = 0;
    while(getline(out, line)) {
        labelText[lineNum] = line;
        lineNum++;
    }
    out.close();

    //create input data for different sizes
    input_tensor_224x224 = vxCreateTensor(context, input_num_of_dims, input_dims_data_224x224, VX_TYPE_FLOAT32, 0);
    input_tensor_28x28 = vxCreateTensor(context, input_num_of_dims, input_dims_data_28x28, VX_TYPE_FLOAT32, 0);

    //read an image and resize to correct dimensions -- opencv imread()
    cv::Mat imagenet_input_image, mnist_input_image, input_image_224x224, input_image_28x28;
    if (runMnist) {
        mnist_input_image = cv::imread(mnist_inputFile_str, cv::CV_LOAD_IMAGE_COLOR);
        if (mnist_input_image.empty()) { //check whether the image is loaded or not
            cout << "ERROR : mnist image is empty" << endl;
            return -1;
        }
    }
    if(runAnyImagenet) {
        imagenet_input_image = cv::imread(imagenet_inputFile_str.c_str(), cv::CV_LOAD_IMAGE_COLOR);
        if (imagenet_input_image.empty()) { //check whether the image is loaded or not
        cout << "ERROR : imagenet image is empty" << endl;
        return -1;
        }
    }

    //resizing and preprocessing for mnist
    if (runMnist) {
        cv::Mat grayscale_input_image;
        cv::resize(mnist_input_image, input_image_28x28, Size(28, 28));
        cv::cvtColor(input_image_28x28, grayscale_input_image, COLOR_BGR2GRAY);

        int rows_grayscale = grayscale_input_image.rows; int cols_grayscale = grayscale_input_image.cols; 
        int total_grayscale = grayscale_input_image.total() * grayscale_input_image.channels();
        unsigned char *input_image_vector_grayscale = (grayscale_input_image.data);

        float *buf_grayscale = (float *)malloc(total_grayscale*sizeof(float));
        float *oneChannel = buf_grayscale;

        float preproc_mul_grayscale = 1;
        
        for(int i = 0; i < rows_grayscale * cols_grayscale; i++, input_image_vector_grayscale ++) {
            *oneChannel++ = ((float)input_image_vector_grayscale[0] * preproc_mul_grayscale);
        }

        status = vxMapTensorPatch(input_tensor_28x28, input_num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for mnist" <<  std::endl;
            return -1;
        }
        
        memcpy(ptr, buf_grayscale, total_grayscale * sizeof(float));

        status = vxUnmapTensorPatch(input_tensor_28x28, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for mnist" <<  std::endl;
            return -1;
        }

        free(buf_grayscale);
    }

    //resizing -- for imagenet
    if(runAnyImagenet) {
        int input_width = imagenet_input_image.size().width;
        int input_height = imagenet_input_image.size().height;
        if(input_height > input_width) {
        int dif = input_height - input_width;
        int bar = floor(dif / 2);
        cv::Range rows((bar + (dif % 2)), (input_height - bar));
        cv::Range cols(0, input_width);
        cv::Mat square = imagenet_input_image(rows, cols);
        cv::resize(square, input_image_224x224, cv::Size(224, 224));
        } else if(input_width > input_height) {
        int dif = input_width - input_height;
        int bar = floor(dif / 2);
        cv::Range rows(0, input_height);
        cv::Range cols((bar + (dif % 2)), (input_width - bar));
        cv::Mat square = imagenet_input_image(rows, cols);
        cv::resize(square, input_image_224x224, cv::Size(224, 224));
        } else {
            cv::resize(imagenet_input_image, input_image_224x224, cv::Size(224, 224));
        }

        //preprocess -- imagenet images
        cv::Mat RGB_input_image;
        cv::cvtColor(input_image_224x224, RGB_input_image, cv::COLOR_BGR2RGB);
        
        int rows = RGB_input_image.rows; int cols = RGB_input_image.cols; 
        int total = RGB_input_image.total() * RGB_input_image.channels();
        unsigned char *input_image_vector = (RGB_input_image.data);

        float *buf = (float *)malloc(total*sizeof(float));
        float *R = buf;
        float *G = R + rows * cols;
        float *B = G + rows * cols;

        float mean_vec[3] = {0.485, 0.456, 0.406};
        float stddev_vec[3] = {0.229, 0.224, 0.225};    
        float preproc_mul[3] = { 1 / (255 * stddev_vec[0]), 1 / (255 * stddev_vec[1]), 1 / (255 * stddev_vec[2])};
        float preproc_add[3] = {(mean_vec[0] / stddev_vec[0]), (mean_vec[1] / stddev_vec[1]), (mean_vec[2] / stddev_vec[2])};
        
        for(int i = 0; i < rows * cols; i++, input_image_vector += 3) {
            *R++ = ((float)input_image_vector[0] * preproc_mul[0]) - preproc_add[0]; 
            *G++ = ((float)input_image_vector[1] * preproc_mul[1]) - preproc_add[1]; 
            *B++ = ((float)input_image_vector[2] * preproc_mul[2]) - preproc_add[2]; 
        }

        ERROR_CHECK_STATUS(vxMapTensorPatch(input_tensor_224x224, input_num_of_dims, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for imagenet" << std::endl;
            return status;
        }

        memcpy(ptr, buf, total * sizeof(float));

        status = vxUnmapTensorPatch(input_tensor_224x224, map_id);
        if (status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for imagenet" << status << ")" << std::endl;
            return status;
        }

        free(buf);
    }

    if (runMnist) {
        //output tensor
        vx_tensor output_tensor_mnist = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x10, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_mnist = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_mnist);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for mnist failed (%d)\n", status);
            return -1;
        }
        vx_node node_mnist = amdMIGraphXnode(graph_mnist, binaryFilename_mnist_str.c_str(), input_tensor_28x28, output_tensor_mnist);
        ERROR_CHECK_OBJECT(node_mnist);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_mnist));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_mnist));

        //mnist timing for 1000 iterations
        t0 = clockCounter();
        for (int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_mnist));
        }
        t1 = clockCounter();
        float mnistTime = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: mnist took %.3f msec (average over %d iterations)\n", mnistTime, N);
        
        //results mnist
        auto num_results_mnist = 10;
        status = vxMapTensorPatch(output_tensor_mnist, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_mnist = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_mnist));
        std::string output_label_mnist = labelText[ID_mnist];

        status = vxUnmapTensorPatch(output_tensor_mnist, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        std::cout << "mnist: output index = " << ID_mnist << std::endl;

        //release resources -- mnist
        ERROR_CHECK_STATUS(vxReleaseNode(&node_mnist));    
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_mnist));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_mnist));
        ERROR_CHECK_STATUS(vxReleaseTensor(&input_tensor_28x28));
    }

    if (runResnet50) {
        //output tensor
        vx_tensor output_tensor_resnet50 = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);
        
        //graph creation
        vx_graph graph_resnet50 = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_resnet50);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for renset50 failed (%d)\n", status);
            return -1;
        }
        vx_node node_resnet50 = amdMIGraphXnode(graph_resnet50, binaryFilename_resnet50_str.c_str(), input_tensor_224x224, output_tensor_resnet50);
        ERROR_CHECK_OBJECT(node_resnet50);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_resnet50));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_resnet50));

        //renet50 timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_resnet50));
        }
        t1 = clockCounter();
        float resnet50Time = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: resnet50 took %.3f msec (average over %d iterations)\n", resnet50Time, N);

        //resnet50 results
        status = vxMapTensorPatch(output_tensor_resnet50, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_resnet50 = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_resnet50 = labelText[ID_resnet50];

        status = vxUnmapTensorPatch(output_tensor_resnet50, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        std::cout << "resnet50: output index = " << ID_resnet50 << "  && output label = " << output_label_resnet50 << std::endl;

        ERROR_CHECK_STATUS(vxReleaseNode(&node_resnet50));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_resnet50));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_resnet50));
    }

    if (runVgg19) {
        //output tensor
        vx_tensor output_tensor_vgg19 = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_vgg19 = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_vgg19);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for vgg19 failed (%d)\n", status);
            return -1;
        }
        vx_node node_vgg19 = amdMIGraphXnode(graph_vgg19, binaryFilename_vgg19_str.c_str(), input_tensor_224x224, output_tensor_vgg19);
        ERROR_CHECK_OBJECT(node_vgg19);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_vgg19));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_vgg19));

        //vgg19 timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_vgg19));
        }
        t1 = clockCounter();
        float vgg19Time = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: vgg19 took %.3f msec (average over %d iterations)\n", vgg19Time, N);

        //vgg19 results
        status = vxMapTensorPatch(output_tensor_vgg19, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_vgg19 = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_vgg19 = labelText[ID_vgg19];

        status = vxUnmapTensorPatch(output_tensor_vgg19, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        //print results
        std::cout << "vgg19: output index = " << ID_vgg19 << "  && output label = " << output_label_vgg19 << std::endl;

        //release resources
        ERROR_CHECK_STATUS(vxReleaseNode(&node_vgg19));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_vgg19));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_vgg19));
    }

    if (runGooglenet) {
        //output tensor
        vx_tensor output_tensor_googlenet = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_googlenet = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_googlenet);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for googlenet failed (%d)\n", status);
            return -1;
        }
        vx_node node_googlenet = amdMIGraphXnode(graph_googlenet, binaryFilename_googlenet_str.c_str(), input_tensor_224x224, output_tensor_googlenet);
        ERROR_CHECK_OBJECT(node_googlenet);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_googlenet));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_googlenet));

        //googlenet timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_googlenet));
        }
        t1 = clockCounter();
        float googlenetTime = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: googlenet took %.3f msec (average over %d iterations)\n", googlenetTime, N);

        //googlenet results
        status = vxMapTensorPatch(output_tensor_googlenet, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_googlenet = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_googlenet = labelText[ID_googlenet];

        status = vxUnmapTensorPatch(output_tensor_googlenet, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        //print results
        std::cout << "googlenet: output index = " << ID_googlenet << "  && output label = " << output_label_googlenet << std::endl;

        //release resources
        ERROR_CHECK_STATUS(vxReleaseNode(&node_googlenet));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_googlenet));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_googlenet));
    }

    if (runAlexnet) {
        //output tensor
        vx_tensor output_tensor_alexnet = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_alexnet = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_alexnet);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for alexnet failed (%d)\n", status);
            return -1;
        }
        vx_node node_alexnet = amdMIGraphXnode(graph_alexnet, binaryFilename_alexnet_str.c_str(), input_tensor_224x224, output_tensor_alexnet);
        ERROR_CHECK_OBJECT(node_alexnet);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_alexnet));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_alexnet));

        //alexnet timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_alexnet));
        }
        t1 = clockCounter();
        float alexnetTime = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: alexnet took %.3f msec (average over %d iterations)\n", alexnetTime, N);

        //alexnet results
        status = vxMapTensorPatch(output_tensor_alexnet, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_alexnet = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_alexnet = labelText[ID_alexnet];

        status = vxUnmapTensorPatch(output_tensor_alexnet, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        //print results
        std::cout << "alexnet: output index = " << ID_alexnet << "  && output label = " << output_label_alexnet << std::endl;

        //release resources
        ERROR_CHECK_STATUS(vxReleaseNode(&node_alexnet));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_alexnet));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_alexnet));

    }

    if (runSqueezenet) {
        //output tensor
        vx_tensor output_tensor_squeezenet = vxCreateTensor(context, output_num_of_dims_4, output_dims_data_1x1000x1x1, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_squeezenet = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_squeezenet);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for squeezenet failed (%d)\n", status);
            return -1;
        }

        vx_node node_squeezenet = amdMIGraphXnode(graph_squeezenet, binaryFilename_squeezenet_str.c_str(), input_tensor_224x224, output_tensor_squeezenet);
        ERROR_CHECK_OBJECT(node_squeezenet);
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_squeezenet));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_squeezenet));

        //squeezenet timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_squeezenet));
        }
        t1 = clockCounter();
        float squeezenetTime = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: squeezenet took %.3f msec (average over %d iterations)\n", squeezenetTime, N);

        //squeezenet results
        status = vxMapTensorPatch(output_tensor_squeezenet, output_num_of_dims_4, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_squeezenet = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_squeezenet = labelText[ID_squeezenet];

        status = vxUnmapTensorPatch(output_tensor_squeezenet, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }

        //print results
        std::cout << "squeezenet: output index = " << ID_squeezenet << "  && output label = " << output_label_squeezenet << std::endl;

        // release resources   
        ERROR_CHECK_STATUS(vxReleaseNode(&node_squeezenet));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_squeezenet));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_squeezenet));
    }

    if (runDensenet) {
        //output tensor
        vx_tensor output_tensor_densenet = vxCreateTensor(context, output_num_of_dims_4, output_dims_data_1x1000x1x1, VX_TYPE_FLOAT32, 0);

        //graph creation
        vx_graph graph_densenet = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph_densenet);
        if(status) {
            printf("ERROR: vxCreateGraph(...) for densenet failed (%d)\n", status);
            return -1;
        }
        vx_node node_densenet = amdMIGraphXnode(graph_densenet, binaryFilename_densenet_str.c_str(), input_tensor_224x224, output_tensor_densenet);
        ERROR_CHECK_OBJECT(node_densenet);  
        ERROR_CHECK_STATUS(vxVerifyGraph(graph_densenet));
        ERROR_CHECK_STATUS(vxProcessGraph(graph_densenet));

        //densenet timing for 1000 iterations
        t0 = clockCounter();
        for(int i = 0; i < N; i++) {
            ERROR_CHECK_STATUS(vxProcessGraph(graph_densenet));
        }
        t1 = clockCounter();
        float densenetTime = (float)(t1-t0)*1000.0f/(float)freq/(float)N;
        printf("OK: densenet took %.3f msec (average over %d iterations)\n", densenetTime, N);

        //densenet results
        status = vxMapTensorPatch(output_tensor_densenet, output_num_of_dims_4, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
            return status;
        }

        int ID_densenet = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results_imagenet));
        std::string output_label_densenet = labelText[ID_densenet];

        status = vxUnmapTensorPatch(output_tensor_densenet, map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
            return status;
        }
        
        //print results
        std::cout << "densenet: output index = " << ID_densenet << "  && output label = " << output_label_densenet << std::endl;

        //release resources
        ERROR_CHECK_STATUS(vxReleaseNode(&node_densenet));
        ERROR_CHECK_STATUS(vxReleaseGraph(&graph_densenet));
        ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_densenet));
    }

    //release common resources
    if (runAnyImagenet) {
        ERROR_CHECK_STATUS(vxReleaseTensor(&input_tensor_224x224));
    }

    ERROR_CHECK_STATUS(vxReleaseContext(&context));
	
    return 0;
}


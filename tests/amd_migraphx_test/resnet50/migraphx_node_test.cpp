#include "vx_amd_migraphx.h"
#include <cstring>
#include <random>
#include <fstream>
#include <algorithm>
#define MAX_STRING_LENGTH 100

using namespace std;

#if ENABLE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#define CVUI_IMPLEMENTATION

using namespace cv;
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

int main(int argc, char **argv) {

    if(argc < 3) {
        std::cout << "Usage: \n ./migraphx_node_test <path-to-resnet50 ONNX model> <path to image>" << std::endl;
        return -1;
    }
    
    std::string modelFileName = argv[1];
    std::string imageFileName = argv[2];

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // initialize variables
    vx_tensor input_tensor, output_tensor;
    vx_size input_num_of_dims = 4;
    vx_size input_dims[4] = {1, 3, 224, 224}; //input dimensions for the resnet50 model
    vx_size output_num_of_dims = 2;
    vx_size output_dims[2] = {1, 1000}; //output dimensions for the resnet50 model
    vx_size stride[4];
    vx_map_id map_id;
    void *ptr = nullptr;
    vx_status status = 0;

    //imagenet label file
    std::string labelText[1000];
    std::string labelFileName = ("../labels.txt");

    std::string line;
    std::ifstream out(labelFileName);
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

    input_tensor = vxCreateTensor(context, input_num_of_dims, input_dims, VX_TYPE_FLOAT32, 0);
    output_tensor = vxCreateTensor(context, output_num_of_dims, output_dims, VX_TYPE_FLOAT32, 0);

    //read an image and resize to correct dimensions -- opencv imread()
    cv::Mat input_image, input_image_224x224;
    input_image = cv::imread(imageFileName);
    
    //resizing
    int input_width = input_image.size().width;
    int input_height = input_image.size().height;
    if(input_height > input_width) {
      int dif = input_height - input_width;
      int bar = floor(dif / 2);
      cv::Range rows((bar + (dif % 2)), (input_height - bar));
      cv::Range cols(0, input_width);
      cv::Mat square = input_image(rows, cols);
      cv::resize(square, input_image_224x224, cv::Size(224, 224));
    } else if(input_width > input_height) {
      int dif = input_width - input_height;
      int bar = floor(dif / 2);
      cv::Range rows(0, input_height);
      cv::Range cols((bar + (dif % 2)), (input_width - bar));
      cv::Mat square = input_image(rows, cols);
      cv::resize(square, input_image_224x224, cv::Size(224, 224));
    } else {
        cv::resize(input_image, input_image_224x224, cv::Size(224, 224));
    }

    //preprocess
    cv::Mat RGB_input_image;
    cv::cvtColor(input_image_224x224, RGB_input_image, cv::COLOR_BGR2RGB);  // cv::imread reads the image in order BGR. SO need to convert
    
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

    ERROR_CHECK_STATUS(vxMapTensorPatch(input_tensor, input_num_of_dims, nullptr, nullptr, &map_id, stride,
        (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    if (status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed " << std::endl;
        return status;
    }

    memcpy(ptr, buf, total * sizeof(float));

    status = vxUnmapTensorPatch(input_tensor, map_id);
    if (status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << status << ")" << std::endl;
        return status;
    }

    ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_amd_migraphx"));

    vx_node node = amdMIGraphXnode(graph, modelFileName.c_str(), input_tensor, output_tensor);
    ERROR_CHECK_OBJECT(node);

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));
    ERROR_CHECK_STATUS(vxProcessGraph(graph));

    status = vxMapTensorPatch(output_tensor, output_num_of_dims, nullptr, nullptr, &map_id, stride,
        (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
        return status;
    }

    //find the argmax
    auto num_results = 1000;
    int final_argmax_result = std::distance((float*)ptr, std::max_element((float*)ptr, (float*)ptr + num_results));
    std::string output_label = labelText[final_argmax_result];

    std::cout << "output index = " << final_argmax_result << "  && output label = " << output_label << std::endl;

    status = vxUnmapTensorPatch(output_tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
        return status;
    }

    // release resources
    ERROR_CHECK_STATUS(vxReleaseNode(&node));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseTensor(&input_tensor));
    ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    free(buf);
	
    return 0;
}


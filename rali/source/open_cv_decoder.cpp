#include "open_cv_decoder.h"

#if OPENCV_FOUND
int handleError( int status, const char* func_name,
            const char* err_msg, const char* file_name,
            int line, void* userdata )
{
    //TODO: add proper error handling here
    return 0;   
}


CVDecoder::CVDecoder() {
    cv::redirectError(handleError);

}
Decoder::Status CVDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) {
    //TODO: OpenCV seems not able to decode header separately, remove the imdecode from this call if possible and replace it with a proper function for decoding the header only
    return Decoder::Status::UNSUPPORTED;
}

Decoder::Status CVDecoder::decode(unsigned char* input_buffer, size_t input_size,  unsigned char* output_buffer,int desired_width, int desired_height, ColorFormat desired_color) {
    //TODO: Find a way to give create an OpenCV image out of the user's provided input pointer and use that to decode the image to
#if 0
    m_mat_compressed = cv::Mat(1, input_size, CV_8UC1, input_buffer);
    m_mat_orig = cv::imdecode(m_mat_compressed, CV_LOAD_IMAGE_COLOR);
    
    if(m_mat_orig.rows == 0 || m_mat_orig.cols == 0) {
        printf("Could not decode the image\n");
        return Status::CONTENT_DECODE_FAILED;
    }
    int width = m_mat_orig.cols;
    int height = m_mat_orig.rows;
    
    return Status::OK;


    cv::resize(m_mat_orig, m_mat_scaled, cv::Size(desired_width, desired_height), cv::INTER_NEAREST);
#endif
    return Decoder::Status::UNSUPPORTED;
}

CVDecoder::~CVDecoder() {
#if 0
    m_mat_scaled.release();
    m_mat_orig.release();
    m_mat_compressed.release();
#endif
}
#endif

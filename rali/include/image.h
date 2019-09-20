#pragma once

#include <VX/vx.h>
#include <VX/vx_types.h>
#include <vector>
#include <cstring>
#include <array>
#include "device_manager.h"
#include "commons.h"

/*! \brief Converts Rali Memory type to OpenVX memory type
 *
 * @param mem input Rali type 
 * @return the OpenVX type associated with input argument
 */
vx_enum vx_mem_type(RaliMemType mem);
struct Point
{
    unsigned x; // Along the width 
    unsigned y; // Along the height
};

struct ROI {
    Point p1; // Upper left
    Point p2; // bottom right
};


// +-----------------------------------------> X direction
// |  ___________________________________
// |  |   p1(x,y)      |                |
// |  |    +-----------|-----------+    |
// |  |    |           |           |    |
// |  -----------------o-----------------
// |  |    |           |           |    |
// |  |    +-----------|-----------+    | 
// |  |                |        p2(x,y) |
// |  +++++++++++++++++++++++++++++++++++
// |
// V Y directoin

/*! \brief Holds the information about an OpenVX image */
struct ImageInfo
{
    friend struct Image;
    enum class Type
    {
        UNKNOWN = -1,
        REGULAR =0,
        VIRTUAL = 1,
        HANDLE =2
    };
    //! Default constructor,
    /*! initializes memory type to host and batch size to 1 */
    ImageInfo();

    //! Initializer constructor
    ImageInfo(
        unsigned width,
        unsigned height,
        unsigned batch_size,
        unsigned color_planes_count,
        RaliMemType mem_type, 
        RaliColorFormat color_format);
    
    bool operator==(const ImageInfo& other);
    unsigned width() { return _width; }
    unsigned height_batch() {return _height * _batch_size; }
    unsigned height_single() { return _height; }
    unsigned color_plane_count() { return _color_planes; }
    void width(unsigned width) { _width = width; }
    void height(unsigned height) { _height = height; }
    Type type() { return _type; }
    const std::string& get_name(unsigned idx) { if(idx < _image_names.size()) return _image_names[idx]; else return _empty_str;}
    unsigned batch_size() {return _batch_size;}
    RaliMemType mem_type() { return _mem_type; }
    unsigned data_size() { return _data_size; }
    RaliColorFormat color_format() {return _color_fmt; }
private:
    Type _type = Type::UNKNOWN;//!< image type, whether is virtual image, created from handle or is a regular image
    unsigned _width;//!< image width for a single image in the batch
    unsigned _height;//!< image height for a single image in the batch
    unsigned _color_planes;//!< number of color planes
    unsigned _batch_size;//!< the batch size (images in the batch are stacked on top of each other)
    unsigned _data_size;//!< total size of the memory needed to keep the image's data in bytes including all planes
    RaliMemType _mem_type;//!< memory type, currently either OpenCL or Host
    RaliColorFormat _color_fmt;//!< color format of the image
    std::vector<std::string> _image_names;//!< image name/ids that are stores in the buffer
    std::string _empty_str;
};
/*! \brief Holds an OpenVX image and it's info 
*
* Keeps the information about the image that can be queried using OVX API as well,
* but for simplicity and ease of use, they are kept in separate fields
*/
struct Image
{
    void* buf = nullptr;//!< Pointer to the image's internal buffer (opencl or host)
    vx_image img = 0;//!< The OpenVX image

    ImageInfo info() { return _info; }
    //! Default constructor 
    Image() = delete;

    unsigned copy_data(unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_mem user_buffer, bool sync);
    void set_names(const std::vector<std::string> names)
    {
        _info._image_names = names;
    }
    //! Default destructor
    /*! Releases the OpenVX image */
    ~Image();

    //! Constructor accepting the image information as input
    Image(const ImageInfo& img_info);

    int create(vx_context context);

    int create_from_handle(vx_context context, ImageBufferAllocation policy);
    void set_command_queue(cl_command_queue queue) { _queue = queue; }
    int create_virtual(vx_context context, vx_graph graph);

private:
    cl_command_queue _queue = nullptr;
    ImageInfo _info;//!< The structure holding the info related to the stored OpenVX image

};
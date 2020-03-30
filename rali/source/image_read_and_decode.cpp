
#include <iterator>
#include <cstring>
#include "decoder_factory.h"
#include "image_read_and_decode.h"



std::tuple<Decoder::ColorFormat, unsigned > 
interpret_color_format(RaliColorFormat color_format ) 
{
    switch (color_format) {
        case RaliColorFormat::RGB24:
            return  std::make_tuple(Decoder::ColorFormat::RGB, 3);

        case RaliColorFormat::BGR24:
            return  std::make_tuple(Decoder::ColorFormat::BGR, 3);

        case RaliColorFormat::U8:
            return  std::make_tuple(Decoder::ColorFormat::GRAY, 1);

        default:
            throw std::invalid_argument("Invalid color format\n");
    }    
}



Timing
ImageReadAndDecode::timing()
{
    Timing t;
    t.image_decode_time = _decode_time.get_timing();
    t.image_read_time = _file_load_time.get_timing();
    return t;
}

ImageReadAndDecode::ImageReadAndDecode():
    _file_load_time("FileLoadTime", DBG_TIMING ),
    _decode_time("DecodeTime", DBG_TIMING)
{
}

ImageReadAndDecode::~ImageReadAndDecode()
{
    _reader = nullptr;
    _decoder.clear();
}   

void
ImageReadAndDecode::create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size)
{
    // Can initialize it to any decoder types if needed
    _batch_size = batch_size;
    _compressed_buff.resize(batch_size);
    _decoder.resize(batch_size);
    _actual_read_size.resize(batch_size);
    _image_names.resize(batch_size);
    _compressed_image_size.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_width.resize(_batch_size);
    _actual_decoded_height.resize(_batch_size);
    for(int i = 0; i < batch_size; i++)
    {
        _compressed_buff[i].resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
        _decoder[i] = create_decoder(decoder_config);
    }
    _reader = create_reader(reader_config);
}

void 
ImageReadAndDecode::reset()
{
    // TODO: Reload images from the folder if needed
    _reader->reset();
}

size_t
ImageReadAndDecode::count()
{
    return _reader->count();
}


LoaderModuleStatus 
ImageReadAndDecode::load(unsigned char* buff,
                         std::vector<std::string>& names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         RaliColorFormat output_color_format )
{
    if(max_decoded_width == 0 || max_decoded_height == 0 )
        THROW("Zero image dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count() < _batch_size)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load images/frames from the disk and push them as a large image onto the buff
    unsigned file_counter = 0;
    const auto ret = interpret_color_format(output_color_format);
    const Decoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);

    // Decode with the height and size equal to a single image  
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing

    while ((file_counter != _batch_size) && _reader->count() > 0)
    {
        size_t fsize = _reader->open();
        if (fsize == 0) {
            WRN("Opened file " + _reader->id() + " of size 0");
            continue;
        }

        _compressed_buff[file_counter].reserve(fsize);

        _actual_read_size[file_counter] = _reader->read(_compressed_buff[file_counter].data(), fsize);
        _image_names[file_counter] = _reader->id();
        _reader->close();
        _compressed_image_size[file_counter] = fsize;
        file_counter++;
    }

    _file_load_time.end();// Debug timing
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);

    for(size_t i = 0; i < _batch_size; i++)
        _decompressed_buff_ptrs[i] = buff + image_size * i;

    _decode_time.start();// Debug timing
#pragma omp parallel for num_threads(_batch_size) default(none)
    for(size_t i= 0; i < _batch_size; i++)
    {
        // initialize the actual decoded height and width with the maximum
        _actual_decoded_width[i] = max_decoded_width;
        _actual_decoded_height[i] = max_decoded_height;
        
        int original_width, original_height, jpeg_sub_samp;
        if(_decoder[i]->decode_info(_compressed_buff[i].data(), _actual_read_size[i], &original_width, &original_height, &jpeg_sub_samp ) != Decoder::Status::OK)
        {
            continue;
        }
#if 0
        if((unsigned)original_width != max_decoded_width || (unsigned)original_height != max_decoded_height)
            // Seeting the whole buffer to zero in case resizing to exact output dimension is not possible.
            memset(_decompressed_buff_ptrs[i],0 , image_size);
#endif

        // decode the image and get the actual decoded image width and height
        size_t scaledw, scaledh;
        if(_decoder[i]->decode(_compressed_buff[i].data(),_compressed_image_size[i],_decompressed_buff_ptrs[i],
                               max_decoded_width, max_decoded_height,
                               original_width, original_height,
                               scaledw, scaledh,
                               decoder_color_format) != Decoder::Status::OK)
        {
            continue;
        }

        _actual_decoded_width[i] = scaledw;
        _actual_decoded_height[i] = scaledh;
    }
    for(size_t i = 0; i < _batch_size; i++)
    {
        names[i] = _image_names[i];
        roi_width[i] = _actual_decoded_width[i];
        roi_height[i] = _actual_decoded_height[i];
    }

    _decode_time.end();// Debug timing

    return LoaderModuleStatus::OK;
}

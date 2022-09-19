/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <iterator>
#include <cstring>
#include "decoder_factory.h"
#include "image_read_and_decode.h"
#include "external_source_reader.h"

std::tuple<Decoder::ColorFormat, unsigned >
interpret_color_format(RocalColorFormat color_format )
{
    switch (color_format) {
        case RocalColorFormat::RGB24:
            return  std::make_tuple(Decoder::ColorFormat::RGB, 3);

        case RocalColorFormat::BGR24:
            return  std::make_tuple(Decoder::ColorFormat::BGR, 3);

        case RocalColorFormat::U8:
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
    t.shuffle_time = _reader->get_shuffle_time();
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
ImageReadAndDecode::create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id)
{
    // Can initialize it to any decoder types if needed
    _batch_size = batch_size;
    _compressed_buff.resize(batch_size);
    _decoder.resize(batch_size);
    _actual_read_size.resize(batch_size);
    _image_names.resize(batch_size); // Check shobi
    _compressed_image_size.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_width.resize(_batch_size);
    _actual_decoded_height.resize(_batch_size);
    _original_height.resize(_batch_size);
    _original_width.resize(_batch_size);
    _decoder_config = decoder_config;
    if ((_decoder_config._type != DecoderType::SKIP_DECODE)) {
        for (int i = 0; i < batch_size; i++) {
            _compressed_buff[i].resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
            _decoder[i] = create_decoder(decoder_config);
            _decoder[i]->initialize(device_id);
        }
    }
    _reader = create_reader(reader_config);
    _reader_type = reader_config.type();
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
    return _reader->count_items();
}

void ImageReadAndDecode::set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader)
{
    _randombboxcrop_meta_data_reader = randombboxcrop_meta_data_reader;
}

std::vector<std::vector<float>>
ImageReadAndDecode::get_batch_random_bbox_crop_coords()
{
    // Return the crop co-ordinates for a batch of images
    return _crop_coords_batch;
}

void
ImageReadAndDecode::set_batch_random_bbox_crop_coords(std::vector<std::vector<float>> crop_coords)
{
    _crop_coords_batch = crop_coords;
}

void ImageReadAndDecode::feed_external_input(std::vector<std::string> input_images, std::vector<int> labels, std::vector<unsigned char *> input_buffer,
                             std::vector<unsigned> roi_width, std::vector<unsigned> roi_height,
                             unsigned int max_width, unsigned int max_height, FileMode mode, bool eos)
{
    std::vector<size_t> image_size; // check if its required
    image_size.reserve(roi_width.size());
    // std::cerr<<"\n roi_width size:: "<<roi_width.size();
    for(unsigned int i = 0; i < roi_width.size(); i++)
        image_size[i] = (roi_width[i] * roi_height[i] * 3); // Shobi check how to get channels
    if(mode == 0) {
        std::cerr<<"\n Mode 0";
        _reader->feed_file_names(input_images, 2, eos); // Shobi check this
    }
    else if(mode == 1) {
        std::cerr<<"\n Mode 1";
        // std::cerr<<"\n Input buffer size :: "<<input_buffer.size();
        // std::cerr<<"\n image_size[n]"<<image_size[0];
        // std::cerr<<"\n width "<<max_width;
        // std::cerr<<"\n height "<<max_height;
        // std::cerr<<"\n channels "<<channels;

        _reader->feed_data(input_buffer, image_size, mode, eos, max_width, max_height, 3);
    }
    else if(mode == 2){
        std::cerr<<"\n Mode 2";
        // std::cerr<<"\n Input buffer size :: "<<input_buffer.size();
        // std::cerr<<"\n image_size[n]"<<image_size[0];
        // std::cerr<<"\n width "<<max_width;
        // std::cerr<<"\n height "<<max_height;
        // std::cerr<<"\n channels "<<channels;

        _reader->feed_data(input_buffer, image_size, mode, eos, max_width, max_height, 3);
    }
    // loader->feed_external_input(input_images, labels, input_buffer, roi_width, roi_height, max_width, max_height, mode);
}

LoaderModuleStatus
ImageReadAndDecode::load(unsigned char* buff,
                         std::vector<std::string>& names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         std::vector<uint32_t> &actual_width,
                         std::vector<uint32_t> &actual_height,
                         RocalColorFormat output_color_format,
                         bool decoder_keep_original )
{
    if(max_decoded_width == 0 || max_decoded_height == 0 )
        THROW("Zero image dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count_items() < _batch_size)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load images/frames from the disk and push them as a large image onto the buff
    unsigned file_counter = 0;
    const auto ret = interpret_color_format(output_color_format);
    const Decoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);
    const bool keep_original = decoder_keep_original;
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);
    bool is_external_source = (_reader_type == StorageType::EXTERNAL_FILE_SOURCE);
    bool skip_decode = false;

    // Decode with the height and size equal to a single image
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing
    if (_decoder_config._type == DecoderType::SKIP_DECODE) {
        while ((file_counter != _batch_size) && _reader->count_items() > 0)
        {
            auto read_ptr = buff + image_size * file_counter;
            size_t fsize = _reader->open();
            if (fsize == 0) {
                WRN("Opened file " + _reader->id() + " of size 0");
                continue;
            }

            _actual_read_size[file_counter] = _reader->read_data(read_ptr, fsize);
            if(_actual_read_size[file_counter] < fsize)
                LOG("Reader read less than requested bytes of size: " + _actual_read_size[file_counter]);

            _image_names[file_counter] = _reader->id();
            _reader->close();
           // _compressed_image_size[file_counter] = fsize;
            names[file_counter] = _image_names[file_counter];
            roi_width[file_counter] = max_decoded_width;
            roi_height[file_counter] = max_decoded_height;
            actual_width[file_counter] = max_decoded_width;
            actual_height[file_counter] = max_decoded_height;
            file_counter++;
        }
        skip_decode = true;
        //_file_load_time.end();// Debug timing
    } else if (is_external_source) {
        auto ext_reader = std::dynamic_pointer_cast<ExternalSourceReader>(_reader);
        if (ext_reader->mode() == FileMode::RAWDATA_UNCOMPRESSED){
        while ((file_counter != _batch_size) && _reader->count_items() > 0) {
              int width, height, channels;
              auto read_ptr = buff + image_size * file_counter;
              size_t fsize = _reader->open();
              if (fsize == 0) {
                  WRN("Opened file " + _reader->id() + " of size 0");
                  continue;
              }

              _actual_read_size[file_counter] = _reader->read_data(read_ptr, fsize);
              if(_actual_read_size[file_counter] < fsize)
                  LOG("Reader read less than requested bytes of size: " + _actual_read_size[file_counter]);

              _image_names[file_counter] = _reader->id();
              ext_reader->get_dims(width, height, channels);

              names[file_counter] = _image_names[file_counter];
              roi_width[file_counter] = width;
              roi_height[file_counter] = height;
              actual_width[file_counter] = width;
              actual_height[file_counter] = height;
              _reader->close();
              file_counter++;
          }
          skip_decode = true;
        }
        else {
            while ((file_counter != _batch_size) && _reader->count_items() > 0) {
                // std::cerr<<"\n ImageReadAndDecode::load CP3";
                size_t fsize = _reader->open();
                // std::cerr<<"\n  ImageReadAndDecode::load CP4  "<<fsize;
                if (fsize == 0) {
                    WRN("Opened file " + _reader->id() + " of size 0");
                    continue;
                }
                // std::cerr<<"\n  ImageReadAndDecode::load CP4  a"<<fsize;
                _compressed_buff[file_counter].reserve(fsize);
                // std::cerr<<"\n  ImageReadAndDecode::load CP4  b"<<fsize;
                _actual_read_size[file_counter] = _reader->read_data(_compressed_buff[file_counter].data(), fsize);
                // std::cerr<<"\n ImageReadAndDecode::load CP5";
                // std::cerr<<"\n file name :: "<<_reader->id();
                _image_names[file_counter] = _reader->id();
                // std::cerr<<"\n ImageReadAndDecode::load CP6";
                _reader->close();
                // std::cerr<<"\n ImageReadAndDecode::load CP7";
                _compressed_image_size[file_counter] = fsize;
                // std::cerr<<"\n ImageReadAndDecode::load CP8";
                file_counter++;
                // std::cerr<<"\n ImageReadAndDecode::load CP9";
            }
        }
    }
    else {
        while ((file_counter != _batch_size) && _reader->count_items() > 0) {
            size_t fsize = _reader->open();
            if (fsize == 0) {
                WRN("Opened file " + _reader->id() + " of size 0");
                continue;
            }
            _compressed_buff[file_counter].reserve(fsize);
            _actual_read_size[file_counter] = _reader->read_data(_compressed_buff[file_counter].data(), fsize);
            _image_names[file_counter] = _reader->id();
            _reader->close();
            _compressed_image_size[file_counter] = fsize;
            file_counter++;
        }

        if (_randombboxcrop_meta_data_reader)
        {
            //Fetch the crop co-ordinates for a batch of images
            _bbox_coords = _randombboxcrop_meta_data_reader->get_batch_crop_coords(_image_names);
            set_batch_random_bbox_crop_coords(_bbox_coords);
        }
    }

    _file_load_time.end();// Debug timing

    _decode_time.start();// Debug timing
    if (!skip_decode) {
        for (size_t i = 0; i < _batch_size; i++)
            _decompressed_buff_ptrs[i] = buff + image_size * i;

// #pragma omp parallel for num_threads(_batch_size)  // default(none) TBD: option disabled in Ubuntu 20.04
        for (size_t i = 0; i < _batch_size; i++)
        {
            // std::cerr<<"\n Decoding image :: "<<i;
            // initialize the actual decoded height and width with the maximum
            _actual_decoded_width[i] = max_decoded_width;
            _actual_decoded_height[i] = max_decoded_height;
            int original_width, original_height, jpeg_sub_samp;
            if (_decoder[i]->decode_info(_compressed_buff[i].data(), _actual_read_size[i], &original_width, &original_height,
                                         &jpeg_sub_samp) != Decoder::Status::OK) {
                    continue;
            }
            _original_height[i] = original_height;
            _original_width[i] = original_width;
            // decode the image and get the actual decoded image width and height
            size_t scaledw, scaledh;
            if(_decoder[i]->is_partial_decoder() && _randombboxcrop_meta_data_reader)
            {
                _decoder[i]->set_bbox_coords(_bbox_coords[i]);
            }
            if (_decoder[i]->decode(_compressed_buff[i].data(), _compressed_image_size[i], _decompressed_buff_ptrs[i],
                                    max_decoded_width, max_decoded_height,
                                    original_width, original_height,
                                    scaledw, scaledh,
                                    decoder_color_format, _decoder_config, keep_original) != Decoder::Status::OK) {
            }
            _actual_decoded_width[i] = scaledw;
            _actual_decoded_height[i] = scaledh;
        }
        // Have to take care of the reserve for vector of strings - shobi
        for (size_t i = 0; i < _batch_size; i++) {
            names[i] = _image_names[i];
            roi_width[i] = _actual_decoded_width[i];
            roi_height[i] = _actual_decoded_height[i];
            actual_width[i] = _original_width[i];
            actual_height[i] = _original_height[i];
            // std::cerr<<"\n Updated image info for image :: "<<names[i];
        }
    }
    _bbox_coords.clear();
    _decode_time.end();// Debug timing
    return LoaderModuleStatus::OK;
}

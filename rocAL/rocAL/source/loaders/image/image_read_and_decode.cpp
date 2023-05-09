/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
    _image_names.resize(batch_size);
    _compressed_image_size.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_width.resize(_batch_size);
    _actual_decoded_height.resize(_batch_size);
    _original_height.resize(_batch_size);
    _original_width.resize(_batch_size);
    _decoder_config = decoder_config;
    _random_crop_dec_param = nullptr;
    if (_decoder_config._type == DecoderType::FUSED_TURBO_JPEG) {
      auto random_aspect_ratio = decoder_config.get_random_aspect_ratio();
      auto random_area = decoder_config.get_random_area();
      AspectRatioRange aspect_ratio_range = std::make_pair((float)random_aspect_ratio[0], (float)random_aspect_ratio[1]);
      AreaRange area_range = std::make_pair((float)random_area[0], (float)random_area[1]);
      _random_crop_dec_param = new RocalRandomCropDecParam(aspect_ratio_range, area_range, (int64_t)decoder_config.get_seed(), decoder_config.get_num_attempts(), _batch_size);
    }
    if ((_decoder_config._type != DecoderType::SKIP_DECODE)) {
        for (int i = 0; i < batch_size; i++) {
            _compressed_buff[i].resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
            _decoder[i] = create_decoder(decoder_config);
            _decoder[i]->initialize(device_id);
        }
    }
    _num_threads = reader_config.get_cpu_num_threads();
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
        //_file_load_time.end();// Debug timing
        //return LoaderModuleStatus::OK;
    } else {
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
        if (_randombboxcrop_meta_data_reader) {
            //Fetch the crop co-ordinates for a batch of images
            _bbox_coords = _randombboxcrop_meta_data_reader->get_batch_crop_coords(_image_names);
            set_batch_random_bbox_crop_coords(_bbox_coords);
        } else if (_random_crop_dec_param) {
            _random_crop_dec_param->generate_random_seeds();
        }
    }

    _file_load_time.end();// Debug timing

    _decode_time.start();// Debug timing
    if (_decoder_config._type != DecoderType::SKIP_DECODE) {
        for (size_t i = 0; i < _batch_size; i++)
            _decompressed_buff_ptrs[i] = buff + image_size * i;

#pragma omp parallel for num_threads(_num_threads)  // default(none) TBD: option disabled in Ubuntu 20.04
        for (size_t i = 0; i < _batch_size; i++)
        {
            // initialize the actual decoded height and width with the maximum
            _actual_decoded_width[i] = max_decoded_width;
            _actual_decoded_height[i] = max_decoded_height;
            int original_width, original_height, jpeg_sub_samp;
            if (_decoder[i]->decode_info(_compressed_buff[i].data(), _actual_read_size[i], &original_width, &original_height,
                                         &jpeg_sub_samp) != Decoder::Status::OK) {
                    // Substituting the image which failed decoding with other image from the same batch
                    int j = ((i + 1) != _batch_size) ? _batch_size - 1 : _batch_size - 2;
                    while ((j >= 0)) 
                    {
                        if (_decoder[i]->decode_info(_compressed_buff[j].data(), _actual_read_size[j], &original_width, &original_height,
                            &jpeg_sub_samp) == Decoder::Status::OK) 
                        {
                                _image_names[i] =  _image_names[j];
                                _compressed_buff[i] =  _compressed_buff[j];
                                _actual_read_size[i] =  _actual_read_size[j];
                                _compressed_image_size[i] =  _compressed_image_size[j];
                                break;                                

                        }
                        else
                            j--;
                        if(j < 0) 
                        {
                            THROW("All images in the batch failed decoding\n");
                        }                                    
                    }
            }
            _original_height[i] = original_height;
            _original_width[i] = original_width;
            // decode the image and get the actual decoded image width and height
            size_t scaledw, scaledh;
            if (_decoder[i]->is_partial_decoder()) {
                if (_randombboxcrop_meta_data_reader) {
                  _decoder[i]->set_bbox_coords(_bbox_coords[i]); 
                } else if (_random_crop_dec_param) {
                  Shape dec_shape = {_original_height[i], _original_width[i]};
                  auto crop_window = _random_crop_dec_param->generate_crop_window(dec_shape, i);
                  _decoder[i]->set_crop_window(crop_window);
                }
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
        for (size_t i = 0; i < _batch_size; i++) {
            names[i] = _image_names[i];
            roi_width[i] = _actual_decoded_width[i];
            roi_height[i] = _actual_decoded_height[i];
            actual_width[i] = _original_width[i];
            actual_height[i] = _original_height[i];
        }
    }
    _bbox_coords.clear();
    _decode_time.end();// Debug timing
    return LoaderModuleStatus::OK;
}

/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "randombboxcrop_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>

//using namespace std;

void RandomBBoxCropReader::init(const RandomBBoxCrop_MetaDataConfig &cfg)
{
    _crop_param = std::make_shared<RaliRandomCropParam>(_batch_size);
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_area_factor(core(cfg.scaling()));
    _crop_param->set_aspect_ratio(core(cfg.aspect_ratio()));
    _all_boxes_overlap = cfg.all_boxes_overlap();
    _no_crop = cfg.no_crop();
    _has_shape = cfg.has_shape();
    _crop_width = cfg.crop_width();
    _crop_height = cfg.crop_height();
    _crop_width_val.resize(_batch_size);
    _crop_height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _x2_val.resize(_batch_size);
    _y2_val.resize(_batch_size);
    _iou_range.resize(_batch_size);
    in_width.resize(_batch_size);
    in_height.resize(_batch_size);
    _crop_param->array_init();
    if(cfg.num_attempts() > 1)
    {
        _num_of_attempts = cfg.num_attempts();
    }
    if(cfg.total_num_attempts() > 0)
    {
        _total_num_of_attempts = cfg.total_num_attempts();
    }
    _output = new CropCordBatch(); 
}

void RandomBBoxCropReader::set_meta_data(std::shared_ptr<MetaDataReader> meta_data_reader)
{
    _meta_data_reader = std::static_pointer_cast<COCOMetaDataReader>(meta_data_reader);
}

bool RandomBBoxCropReader::exists(const std::string &image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

inline double ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false)
{
    double iou;
    float xA = std::max(box1.x, box2.x);
    float yA = std::max(box1.y, box2.y);
    float xB = std::min(box1.x + box1.w, box2.x + box2.w);
    float yB = std::min(box1.y + box1.h, box2.y + box2.h);

    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);

    float box1_area = box1.h * box1.w;
    float box2_area = box2.h * box2.w;

    if (is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void RandomBBoxCropReader::lookup(const std::vector<std::string>& image_names)
{
    if (image_names.empty())
    {
        std::cerr<<"\n No images passed";
        WRN("No image names passed")
        return;
    }if (image_names.size() != (unsigned)_output->size())
    {
        _output->resize(image_names.size());
    }
    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it) 
            THROW("ERROR: Given name not present in the map" + image_name)    
        _output->get_bb_cords_batch()[i] = it->second;
    }
}

pCropCord RandomBBoxCropReader::get_crop_cord(const std::string &image_name)
{
    if (image_name.empty())
    {
        WRN("No image names passed")
        return 0;
    }
    auto it = _map_content.find(image_name);
    if (_map_content.end() == it) 
        THROW("ERROR: Given name not present in the map" + image_name)    
    return it->second;
}

void RandomBBoxCropReader::add(std::string image_name, BoundingBoxCord crop_box)
{
    pCropCord random_bbox_cords = std::make_shared<CropCord>(crop_box.x, crop_box.y, crop_box.w, crop_box.h);
    if (exists(image_name))
    {
        // auto it = _map_content.find(image_name);
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<CropCord>>(image_name,random_bbox_cords));
}

void RandomBBoxCropReader::print_map_contents()
{
    pCropCord random_bbox_cords;

    std::cerr << "\n ********************************Map contents:***************************** \n";
    for (auto& elem : _map_content) {
        std::cerr << "\n Name :\t " << elem.first;
        random_bbox_cords = elem.second;
        std::cerr<<"\n Crop values:: crop_x:: "<<random_bbox_cords->crop_x<<"\t crop_y:: "<<random_bbox_cords->crop_y<<"\t crop_width:: "<<random_bbox_cords->crop_width<<"\t crop_height:: "<<random_bbox_cords->crop_height;
    }
}

void RandomBBoxCropReader::read_all()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 6);

    const std::vector<float> sample_options = {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    int sample_option;
    std::pair<bool, float> option;
    float min_iou; // max_iou;
    bool invalid_bboxes = false;
    _entire_iou = true;
    bool _overlap_iou = false;
    BoundingBoxCord crop_box, jth_box;
    uint bb_count;
    _meta_bbox_map_content = _meta_data_reader->get_map_content();
    int i = 0;
    for (auto& elem : _meta_bbox_map_content)
    {
        std::string image_name =  elem.first;
        BoundingBoxCords bb_coords = elem.second->get_bb_cords();
        ImgSizes img_sizes = elem.second->get_img_sizes();
        in_width[i] = img_sizes[i].w;
        in_height[i] = img_sizes[i].h;
        bb_count = bb_coords.size();
        std::vector<float> coords_buf(bb_count * 4);
        for(unsigned int j = 0, m = 0; j < bb_count; j++, m += 4){
            coords_buf[m] = bb_coords[j].x;
            coords_buf[m + 1] = bb_coords[j].y;
            coords_buf[m + 2] = bb_coords[j].w;
            coords_buf[m + 3] = bb_coords[j].h;
        } 
        _crop_param->set_image_dimensions(in_width, in_height);
        _crop_param->fill_crop_dims();
        Parameter<float> *x_drift_factor = _crop_param->get_x_drift_factor();
        Parameter<float> *y_drift_factor = _crop_param->get_y_drift_factor();
        _x1_val = _crop_param->get_x1_arr_val();
        _y1_val = _crop_param->get_y1_arr_val();
        _crop_width_val = _crop_param->get_cropw_arr_val();
        _crop_height_val = _crop_param->get_croph_arr_val();
        bool crop_success = false;
        
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        if(_total_num_of_attempts == 0)
            _total_num_of_attempts = 10;
        bool TrueFalse = (rand() % 100) < 25; //50 is the probability
        // Got BBOX Information of the image, try to get a crop
        //Crop the Image if TrueFalse is 1, else , Keep Image as it is
        if (TrueFalse)
        {
             int count = 0;
            while (!crop_success && (_total_num_of_attempts == 0 || count < _total_num_of_attempts))
            {
                sample_option = dis(gen);
                min_iou =sample_options[sample_option];
                if (_has_shape)
                {
                    // std::cerr<<"\n Coming to has_shape";
                    crop_box.w = _crop_width - 1;  // Given By user
                    crop_box.h = _crop_height - 1; // Given By user
                }
                else // If it has no shape, then area and aspect ratio thing should be provided
                {
                    for (int j = 0; j < _num_of_attempts; j++)
                    {
                        count++;
                        x_drift_factor->renew(); //_scale_factor is a random_number picked from [min-max] range
                        auto w_factor = x_drift_factor->get();
                        crop_box.w = w_factor * in_width[i];
                        y_drift_factor->renew(); //_scale_factor is a random_number picked from [min-max] range
                        y_drift_factor->renew();
                        auto h_factor = y_drift_factor->get();
                        crop_box.h = h_factor * in_height[i];
                        if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                        {
                            continue;
                        }
                        break;
                    }
                    if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                    {
                        continue;
                    }
                }

                x_drift_factor->renew();
                x_drift_factor->renew();
                y_drift_factor->renew(); // Get the random parameters between 0 - 1 (Uniform random)
                crop_box.x = static_cast<size_t>(x_drift_factor->get() * (in_width[i] - crop_box.w));
                crop_box.y = static_cast<size_t>(y_drift_factor->get() * (in_height[i] - crop_box.h));
                bool entire_iou = !_overlap_iou; //
                if (_all_boxes_overlap)
                {
                    for (uint j = 0; j < bb_count; j++)
                    {
                        int m = j * 4;
                        jth_box.x = coords_buf[m];
                        jth_box.y = coords_buf[m + 1];
                        jth_box.w = coords_buf[m + 2];
                        jth_box.h = coords_buf[m + 3];
                        float bb_iou = ssd_BBoxIntersectionOverUnion(jth_box, crop_box, entire_iou);
                        if (bb_iou < min_iou)
                        {
                            invalid_bboxes = true;
                            break;
                        }
                    }
                    if (invalid_bboxes)
                    {
                        continue;
                    }
                }
                else // at lease one box shoud overlap
                {
                    invalid_bboxes = true;
                    for (uint j = 0; j < bb_count; j++)
                    {
                        int m = j * 4;
                        jth_box.x = coords_buf[m];
                        jth_box.y = coords_buf[m + 1];
                        jth_box.w = coords_buf[m + 2];
                        jth_box.h = coords_buf[m + 3];
                        float bb_iou = ssd_BBoxIntersectionOverUnion(jth_box, crop_box, entire_iou);
                        if (bb_iou > min_iou)
                        {
                            invalid_bboxes = false;
                            break;
                        }
                    }
                    if (invalid_bboxes)
                    {
                        continue;
                    }
                }

                int valid_bbox_count = 0;
                auto left = crop_box.x, top = crop_box.y, right = crop_box.x + crop_box.w, bottom = crop_box.y + crop_box.h;
                for (uint j = 0; j < bb_count; j++)
                {
                    int m = j * 4;
                    auto x_c = 0.5f * (2 * coords_buf[m] + coords_buf[m + 2]);
                    auto y_c = 0.5f * (2 * coords_buf[m + 1] + coords_buf[m + 3]);
                    if ((x_c >= left) && (x_c <= right) && (y_c >= top) && (y_c <= bottom))
                        valid_bbox_count++;
                }
                if (valid_bbox_count == 0)
                    continue;
                crop_success = true;
            } // while loop
        }
        if(!crop_success && _no_crop)
        {
            crop_box.x = 0;
            crop_box.y = 0;
            crop_box.w = in_width[i];
            crop_box.h = in_height[i];
        }
        add(image_name, crop_box);
    }
}

void RandomBBoxCropReader::release()
{
    _map_content.clear();
}

RandomBBoxCropReader::RandomBBoxCropReader()
{
}

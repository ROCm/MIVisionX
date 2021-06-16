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
#include "rali_api.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>

void RandomBBoxCropReader::init(const RandomBBoxCrop_MetaDataConfig &cfg)
{
    _all_boxes_overlap = cfg.all_boxes_overlap();
    _no_crop = cfg.no_crop();
    _has_shape = cfg.has_shape();
    if (cfg.num_attempts() > 1)
    {
        _num_of_attempts = cfg.num_attempts();
    }
    if (cfg.total_num_attempts() > 0)
    {
        _total_num_of_attempts = cfg.total_num_attempts();
    }
    _output = new CropCordBatch();
    _user_batch_size = 128;   // todo:: get it from master graph
    _seed = cfg.seed();
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
    float xA = std::max(box1.l, box2.l);
    float yA = std::max(box1.t, box2.t);
    float xB = std::min(box1.r, box2.r);
    float yB = std::min(box1.b, box2.b);

    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    float box1_area = (box1.b - box1.t) * (box1.r - box1.l);
    float box2_area = (box2.b - box2.t) * (box2.r - box2.l);

    if (is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void RandomBBoxCropReader::lookup(const std::vector<std::string> &image_names)
{
    if (image_names.empty())
    {
        std::cerr << "\n No images passed";
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
    {
        _output->resize(image_names.size());
    }
    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        // std::cerr<<"\n Master Graph lookup:: "<<image_name;
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
    
    pCropCord random_bbox_cords = std::make_shared<CropCord>(crop_box.l, crop_box.t, crop_box.r , crop_box.b );
    if (exists(image_name))
    {
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<CropCord>>(image_name, random_bbox_cords));
}

void RandomBBoxCropReader::print_map_contents()
{
    pCropCord random_bbox_cords;

    std::cerr << "\n ********************************Map contents:***************************** \n";
    for (auto &elem : _map_content)
    {
        std::cerr << "\n Name :\t " << elem.first;
        random_bbox_cords = elem.second;
        std::cerr << "\n Crop values:: crop_left:: " << random_bbox_cords->crop_left << "\t crop_top:: " << random_bbox_cords->crop_top << "\t crop_right:: " << random_bbox_cords->crop_right << "\t crop_bottom:: " << random_bbox_cords->crop_bottom;
    }
}

void RandomBBoxCropReader::read_all()
{
    const std::vector<float> sample_options = {-1.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.0f};
    int sample_option;
    std::pair<bool, float> option;
    float min_iou;
    bool invalid_bboxes;
    bool crop_success;
    BoundingBoxCord crop_box;
    uint bb_count;
    _meta_bbox_map_content = _meta_data_reader->get_map_content();
    std::uniform_int_distribution<> option_dis(0, 6);
    std::uniform_real_distribution<float> _float_dis(0.3, 1.0);

    size_t sample = 0;
    for (auto &elem : _meta_bbox_map_content)
    {
        
        std::string image_name = elem.first;
        BoundingBoxCords bb_coords = elem.second->get_bb_cords();
        ImgSizes img_sizes = elem.second->get_img_sizes();
        bb_count = bb_coords.size();
        while (true)
        {
            crop_success = false;
            sample_option = option_dis(_rngs[sample]);
            min_iou = sample_options[sample_option];
            invalid_bboxes = false;

            //Condition for Original Image
            if (sample_option == 6 || _has_shape)
            {
                crop_box.l = 0;
                crop_box.t = 0;
                crop_box.r = 1;
                crop_box.b = 1;
                break;
            }

            // If it has no shape, then area and aspect ratio thing should be provided
            for (int j = 0; j < 1; j++)
            {
                // Setting width and height factor btw 0.3 and 1.0";
                float width_factor = _float_dis(_rngs[sample]);
                float height_factor = _float_dis(_rngs[sample]);
                if ((width_factor / height_factor < 0.5) || (width_factor / height_factor > 2.))
                {
                    continue;
                }
                // Setting width factor btw 0 and 1 - width_factor and height factor btw 0 and 1 - height_factor
                std::uniform_real_distribution<float> l_dis(0.0, 1.0 - width_factor), t_dis(0.0, 1.0-height_factor);
                float x_factor = l_dis(_rngs[sample]);
                float y_factor = t_dis(_rngs[sample]);
                crop_box.l = x_factor;
                crop_box.t = y_factor;
                crop_box.r = crop_box.l + width_factor;
                crop_box.b = crop_box.t + height_factor;
                //std::cout << "random crop params < option, xfactor, yfactor, wf, hf>: " << sample_option << " " << x_factor << " " << y_factor << " " << width_factor << " " << height_factor << std::endl;
                // All boxes should satisfy IOU criteria
                if (_all_boxes_overlap)
                {
                    for (uint j = 0; j < bb_count; j++)
                    {
                        float bb_iou = ssd_BBoxIntersectionOverUnion(bb_coords[j], crop_box, true);
                        if (bb_iou < min_iou)
                        {
                            invalid_bboxes = true;
                            break;
                        }
                    }
                    if (invalid_bboxes)
                    {
                        continue; // Goes to for loop
                    }
                }

                // Mask Condition
                int valid_bbox_count = 0;
                valid_bbox_count = 0;
                for (uint j = 0; j < bb_count; j++)
                {
                    auto x_c = 0.5 * (bb_coords[j].l + bb_coords[j].r);
                    auto y_c = 0.5 * (bb_coords[j].t + bb_coords[j].b);
                    if ((x_c > crop_box.l) && (x_c < crop_box.r) && (y_c > crop_box.t) && (y_c < crop_box.b))
                    {
                        valid_bbox_count++;
                    }
                }
                if (valid_bbox_count == 0)
                    break;

                crop_success = true;
                break;
            }

            if (crop_success == true)
                break;
        } // while loop

        //        std::cout << image_name << " wxh: " << in_width << "X" << in_height << " crop<x,y, w, h>: " << crop_box.x << " X " << crop_box.y << " X " << crop_box.w << " X " << crop_box.h << std::endl;
        add(image_name, crop_box);
        sample++;
    }
}

void RandomBBoxCropReader::release()
{
    _map_content.clear();
}

RandomBBoxCropReader::RandomBBoxCropReader() :
      _rngs(128)
{
}

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

#include "tf_meta_data_reader_detection.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include<fstream>
#include <string>
#include <stdint.h>
#include <google/protobuf/message_lite.h>
#include "example.pb.h"
#include "feature.pb.h"

using namespace std;

void TFMetaDataReaderDetection::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new BoundingBoxBatch();
    _last_rec = false;
}

bool TFMetaDataReaderDetection::exists(const std::string& _image_name)
{
    return _map_content.find(_image_name) != _map_content.end();
}

// void TFMetaDataReaderDetection::add(std::string _image_name, int label)
// {
//     pMetaData info = std::make_shared<Label>(label);
//     if(exists(_image_name))
//     {
//         WRN("Entity with the same name exists")
//         return;
//     }
//     _map_content.insert(pair<std::string, std::shared_ptr<Label>>(_image_name, info));
// }

void TFMetaDataReaderDetection::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels)
{
    if(exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void TFMetaDataReaderDetection::lookup(const std::vector<std::string> &image_names)
{
    
    
    if(image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())   
        _output->resize(image_names.size());

    for(unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
	/*
	 * User should provide the coco train or val folder containing images with respect to json file.
	 * If the processed COCO image was not in the map, returns BoundingBox meta data values as zero since 
	 * those images doesn't have annotations.
	 */
        if(_map_content.end() == it){

            BoundingBoxCords bb_coords;
            BoundingBoxLabels bb_labels;
            BoundingBoxCord box;

            box.x = box.y = box.w = box.h = 0;
            bb_coords.push_back(box);
            bb_labels.push_back(0);
            // bb_coords={};

	    _output->get_bb_cords_batch()[i] = bb_coords;
	    _output->get_bb_labels_batch()[i] = bb_labels;
        }
	else{
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
	}
    }




    

}

void TFMetaDataReaderDetection::print_map_contents()
{
    // std::cerr << "\nMap contents: \n";
    // for (auto& elem : _map_content) {
    //     std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    // }

    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;

    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords() ;
        bb_labels = elem.second->get_bb_labels();
        std::cerr << "\nsize of the element  : "<< bb_coords.size() << std::endl;
        for(unsigned int i = 0; i < bb_coords.size(); i++){
            std::cerr << " x : " << bb_coords[i].x << " y: :" << bb_coords[i].y << " width : " << bb_coords[i].w << " height: :" << bb_coords[i].h << std::endl;
            std::cerr  << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void TFMetaDataReaderDetection::read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &_image_name)
{
    uint length;
    length = file_contents.tellg();

    std::string temp;
    size_t uint64_size, uint32_size;
    uint64_t data_length;
    uint32_t length_crc, data_crc;
    uint64_size = sizeof(uint64_t); 
    uint32_size = sizeof(uint32_t); 
    char * header_length = new char [uint64_size];
    char * header_crc = new char [uint32_size];
    char * footer_crc = new char [uint32_size];
    file_contents.read(header_length, uint64_size);
    file_contents.read(header_crc, uint32_size);
    memcpy(&data_length, header_length, sizeof(data_length));
    memcpy(&length_crc, header_crc, sizeof(length_crc));

    if(length + data_length + 16 == file_size){
        _last_rec = true;
    }
    char *data = new char[data_length];
    file_contents.read(data,data_length);
    tensorflow::Example single_example;
    single_example.ParseFromArray(data,data_length);
    tensorflow::Features features = single_example.features();

//..............
    // tensorflow::Features features = single_example.features();
    // features.PrintDebugString();
//..............

    auto feature = features.feature();
    tensorflow::Feature single_feature,sf_xmin,sf_ymin,sf_xmax,sf_ymax,sf_fname,sf_label;
    
    single_feature = feature.at("image/filename");
    std::string fname = single_feature.bytes_list().value()[0];
    
    
    float bbox_xmin,bbox_ymin,size_b_xmin,bbox_xmax,bbox_ymax;
    single_feature = feature.at("image/object/bbox/xmin");
    size_b_xmin = single_feature.float_list().value().size();
    

    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    BoundingBoxCord box;

    

    int label;
    single_feature = feature.at("image/class/label");
    label = single_feature.int64_list().value()[0];

    sf_xmin = feature.at("image/object/bbox/xmin");
    sf_ymin = feature.at("image/object/bbox/ymin");
    sf_xmax = feature.at("image/object/bbox/xmax");
    sf_ymax = feature.at("image/object/bbox/ymax");
    
    for(int i=0;i<size_b_xmin;i++)
    {
      
      bbox_xmin = sf_xmin.float_list().value()[i];
      bbox_ymin = sf_ymin.float_list().value()[i];
      bbox_xmax = sf_xmax.float_list().value()[i];
      bbox_ymax = sf_ymax.float_list().value()[i];
       
      box.x = bbox_xmin;
      box.y = bbox_ymin;
      box.w = bbox_xmax;
      box.h = bbox_ymax;
      bb_coords.push_back(box);
      bb_labels.push_back(label);
      add(fname, bb_coords, bb_labels);
      bb_coords.clear();
      bb_labels.clear();
    }


    file_contents.read(footer_crc, sizeof(data_crc));
    memcpy(&data_crc, footer_crc, sizeof(data_crc));
    free(header_length);
    free(header_crc);
    free(footer_crc);
    free(data);
}

void TFMetaDataReaderDetection::read_all(const std::string &path)
{
    read_files(path);
    for(unsigned i = 0; i < _file_names.size(); i++)
    {
        std::string fname = path + _file_names[i];
        uint length;
        std::cerr<< "file_name:: "<<fname<<std::endl;
        std::ifstream file_contents(fname.c_str(),std::ios::binary);
        file_contents.seekg (0, std::ifstream::end);
        length = file_contents.tellg();
        file_contents.seekg (0, std::ifstream::beg);
        while(!_last_rec)
        {
            read_record(file_contents, length, _image_name);
        }
        _last_rec = false;
        file_contents.close();
    }
  //google::protobuf::ShutdownProtobufLibrary();
    print_map_contents();

}

void TFMetaDataReaderDetection::release(std::string _image_name)
{
    if(!exists(_image_name))
    {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void TFMetaDataReaderDetection::release() {
    _map_content.clear();
}

void TFMetaDataReaderDetection::read_files(const std::string& _path)
{
    if ((_src_dir = opendir (_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        _file_names.push_back(_entity->d_name);  
    }
    if(_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}


TFMetaDataReaderDetection::TFMetaDataReaderDetection()
{
}
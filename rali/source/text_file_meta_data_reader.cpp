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

#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include "commons.h"
#include "exception.h"
#include "text_file_meta_data_reader.h"

void TextFileMetaDataReader::init(const MetaDataConfig &cfg) {
	_path = cfg.path();
    _output = new LabelBatch();
}

bool TextFileMetaDataReader::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void TextFileMetaDataReader::add(std::string image_name, int label)
{
    pMetaData info = std::make_shared<Label>(label);
    if(exists(image_name))
    {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void TextFileMetaDataReader::lookup(const std::vector<std::string> &image_names) {
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
        if(_map_content.end() == it)
            THROW("ERROR: Given name not present in the map"+ image_name )
        _output->get_label_batch()[i] = it->second->get_label();
    }
}

void TextFileMetaDataReader::read_all(const std::string &path) {
	std::ifstream text_file(path.c_str());
	if(text_file.good())
	{
		//_text_file.open(path.c_str(), std::ifstream::in);
		std::string line;
		while(std::getline(text_file, line))
		{
            std::istringstream line_ss(line);
            int label;
            std::string image_name;
            if(!(line_ss>>image_name>>label))
                continue;
			add(image_name, label);
		}
	}
	else
    {
	    THROW("Can't open the metadata file at "+ path)
    }
}

void TextFileMetaDataReader::release(std::string image_name) {
	if(!exists(image_name))
    {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void TextFileMetaDataReader::release() {
	_map_content.clear();
}

TextFileMetaDataReader::TextFileMetaDataReader() {

}
//
// Created by mvx on 3/31/20.
//


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

#pragma once
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "commons.h"


typedef  struct { float x; float y; float w; float h; } BoundingBoxCord;
typedef  std::vector<BoundingBoxCord> BoundingBoxCords;
typedef  std::vector<int> BoundingBoxLabels;
typedef  struct { int w; int h; } ImgSize;
typedef  std::vector<ImgSize> ImgSizes;

struct MetaData
{
    int& get_label() { return _label_id; }
    BoundingBoxCords& get_bb_cords() { return _bb_cords; }
    BoundingBoxLabels& get_bb_labels() { return _bb_label_ids; }
    ImgSizes& get_img_sizes() {return _img_sizes; }
protected:
    BoundingBoxCords _bb_cords = {}; // For bb use
    BoundingBoxLabels _bb_label_ids = {};// For bb use
    ImgSizes _img_sizes = {};
    int _label_id = -1; // For label use only
};

struct Label : public MetaData
{
    Label(int label) { _label_id = label; }
    Label(){ _label_id = -1; }
};

struct BoundingBox : public MetaData
{
    BoundingBox()= default;
    BoundingBox(BoundingBoxCords bb_cords,BoundingBoxLabels bb_label_ids )
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords bb_cords,BoundingBoxLabels bb_label_ids ,ImgSizes img_sizes)
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
        _img_sizes = std::move(img_sizes);
    }
    void set_bb_cords(BoundingBoxCords bb_cords) { _bb_cords =std::move(bb_cords); }
    void set_bb_labels(BoundingBoxLabels bb_label_ids) {_bb_label_ids = std::move(bb_label_ids); }
    void set_img_sizes(ImgSizes img_sizes) { _img_sizes =std::move(img_sizes); }
};

struct MetaDataBatch
{
    virtual ~MetaDataBatch() = default;
    virtual void clear() = 0;
    virtual void resize(int batch_size) = 0;
    virtual int size() = 0;
    virtual MetaDataBatch&  operator += (MetaDataBatch& other) = 0;
    MetaDataBatch* concatenate(MetaDataBatch* other)
    {
        *this += *other;
        return this;
    }
    virtual std::shared_ptr<MetaDataBatch> clone()  = 0;
    std::vector<int>& get_label_batch() { return _label_id; }
    std::vector<BoundingBoxCords>& get_bb_cords_batch() { return _bb_cords; }
    std::vector<BoundingBoxLabels>& get_bb_labels_batch() { return _bb_label_ids; }
    std::vector<ImgSizes>& get_img_sizes_batch() { return _img_sizes; }
protected:
    std::vector<int> _label_id = {}; // For label use only
    std::vector<BoundingBoxCords> _bb_cords = {};
    std::vector<BoundingBoxLabels> _bb_label_ids = {};
    std::vector<ImgSizes> _img_sizes = {};
};

struct LabelBatch : public MetaDataBatch
{
    void clear() override
    {
        _label_id.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _label_id.insert(_label_id.end(),other.get_label_batch().begin(), other.get_label_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _label_id.resize(batch_size);
    }
    int size() override
    {
        return _label_id.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<LabelBatch>(*this);
    }
    explicit LabelBatch(std::vector<int>& labels)
    {
        _label_id = std::move(labels);
    }
    LabelBatch() = default;
};

struct BoundingBoxBatch: public MetaDataBatch
{
    void clear() override
    {
        _bb_cords.clear();
        _bb_label_ids.clear();
        _img_sizes.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(),other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _bb_label_ids.insert(_bb_label_ids.end(), other.get_bb_labels_batch().begin(), other.get_bb_labels_batch().end());
        _img_sizes.insert(_img_sizes.end(),other.get_img_sizes_batch().begin(), other.get_img_sizes_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
        _img_sizes.resize(batch_size);
    }
    int size() override
    {
        return _bb_cords.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<BoundingBoxBatch>(*this);
    }
};
using ImageNameBatch = std::vector<std::string>;
using pMetaData = std::shared_ptr<Label>;
using pMetaDataBox = std::shared_ptr<BoundingBox>;
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;

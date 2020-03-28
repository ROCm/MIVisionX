#pragma once
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "commons.h"


typedef  struct { int x1; int x2; int y1; int y2; } BoundingBoxCord;
typedef  std::vector<BoundingBoxCord> BoundingBoxCords;
typedef  std::vector<int> BoundingBoxLabels;

struct MetaData
{
    int& get_label() { return _label_id; }
    BoundingBoxCords& get_bb_cords() { return _bb_cords; }
    BoundingBoxLabels& get_bb_labels() { return _bb_label_ids; }
protected:
    BoundingBoxCords _bb_cords = {}; // For bb use
    BoundingBoxLabels _bb_label_ids = {};// For bb use
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
    void set_bb_cords(BoundingBoxCords bb_cords) { _bb_cords =std::move(bb_cords); }
    void set_bb_labels(BoundingBoxLabels bb_label_ids) {_bb_label_ids = std::move(bb_label_ids); }
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
protected:
    std::vector<int> _label_id = {}; // For label use only
    std::vector<BoundingBoxCords> _bb_cords = {};
    std::vector<BoundingBoxLabels> _bb_label_ids = {};
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
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(),other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _bb_label_ids.insert(_bb_label_ids.end(), other.get_bb_labels_batch().begin(), other.get_bb_labels_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
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
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <dirent.h>
#include "reader.h"
#include "meta_data_reader.h"
#include "meta_data_graph.h"
#include "timing_debug.h"



class COCOFileSourceReader : public Reader {
public:
    //! Looks up the folder which contains the files, amd loads the image names
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id;};

    unsigned count() override;
    unsigned long long get_shuffle_time() {return _shuffle_time.get_timing();};

    ~COCOFileSourceReader() override;

    int close() override;

    COCOFileSourceReader();

private:
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    //! opens the folder containnig the images
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    std::string _json_path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _files;
    unsigned  _curr_file_idx;
    FILE* _current_fPtr;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    size_t _shard_id = 0;
    size_t _shard_count = 1;// equivalent of batch size
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t  _file_count_all_shards;
    void incremenet_read_ptr();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void replicate_last_batch_to_pad_partial_shard();
    TimingDBG _shuffle_time;
};

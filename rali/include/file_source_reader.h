#pragma once
#include <vector>
#include <string>
#include <memory>
#include <dirent.h>
#include "reader.h"



class FileSourceReader : public Reader {
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
    
    ~FileSourceReader() override;

    int close() override;

    FileSourceReader();

private:
    //! opens the folder containnig the images
    Reader::Status open_folder();
    std::string _folder_path;
    DIR *_src_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    unsigned  _curr_file_idx;
    FILE* _current_fPtr;
    unsigned _current_file_size;
    std::string _last_id;
    size_t _read_offset = 0;
    size_t _read_interval = 1;
    bool _loop;
    int _read_counter = 0;
    void incremenet_read_ptr();

    int release();
};
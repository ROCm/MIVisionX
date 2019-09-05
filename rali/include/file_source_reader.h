#pragma once
#include <vector>
#include <string>
#include <memory>
#include <dirent.h>
#include "reader.h"


class FileSourceReaderConfig : public ReaderConfig {
    public:
    explicit FileSourceReaderConfig(const std::string& folder_path_, size_t read_offset_ = 0, size_t read_interval_ = 1): 
    folder_path(folder_path_), 
    read_interval(read_interval_), 
    read_offset(read_offset_) 
    {}
    ReaderType type() override { return ReaderType::FILE_SOURCE;}
    std::string folder_path;
    size_t read_interval;
    size_t read_offset;
};

class FileSourceReader : public Reader {
public:
    //! Looks up the folder which contains the files, amd loads the image names
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig* desc);
    
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

    unsigned count();
    
    ~FileSourceReader();

    int close();

    FileSourceReader();

private:
    //! opens the folder containnig the images
    Reader::Status open_folder();
    std::string m_folder_path;
    DIR *m_src_dir;
    struct dirent *m_entity;
    std::vector<std::string> m_file_names;
    int m_curr_file_idx;
    FILE* m_current_fPtr;
    unsigned m_current_file_size;
    std::string _last_id;
    size_t _read_offset = 0;
    size_t _read_interval = 1;
};
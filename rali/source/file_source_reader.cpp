#include <cassert>
#include <commons.h>
#include "file_source_reader.h"

FileSourceReader::FileSourceReader() 
{
    m_src_dir = nullptr;
    m_entity = nullptr;
    m_curr_file_idx = 0;
    m_current_file_size = 0;
    m_current_fPtr = nullptr;
}

unsigned FileSourceReader::count() 
{
    int ret = ((int)m_file_names.size() - m_curr_file_idx);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status FileSourceReader::initialize(ReaderConfig* desc) 
{
    auto file_reader_desc = dynamic_cast<FileSourceReaderConfig*>(desc);
    m_folder_path = file_reader_desc->folder_path;
    _read_offset = file_reader_desc->read_offset;
    _read_interval = file_reader_desc->read_interval;
    return open_folder();
}

size_t FileSourceReader::open() 
{
    auto file_path = m_file_names[m_curr_file_idx++];// Get next file name 

    _last_id= file_path;
    
    m_current_fPtr = fopen(file_path.c_str(), "rb");// Open the file, 
    
    if(!m_current_fPtr) // Check if it is ready for reading
        return 0;
    
    fseek(m_current_fPtr, 0 , SEEK_END);// Take the file read pointer to the end

    m_current_file_size = ftell(m_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)
    
    if(m_current_file_size == 0) 
    { // If file is empty continue
        fclose(m_current_fPtr);
        m_current_fPtr = 0;
        return 0;
    }
    
    fseek(m_current_fPtr, 0 , SEEK_SET);// Take the file pointer back to the start

    return m_current_file_size;
}

size_t FileSourceReader::read(unsigned char* buf, size_t read_size)
{
    if(!m_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > m_current_file_size) ? m_current_file_size: read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, m_current_fPtr);
    return actual_read_size;
}

int FileSourceReader::close() 
{
    if(!m_current_fPtr)
        return 0;
    fclose(m_current_fPtr);
    m_current_fPtr = 0;
    return 0;
}

FileSourceReader::~FileSourceReader() 
{
    close();
}


void FileSourceReader::reset() 
{
    m_curr_file_idx = 0;
}

Reader::Status FileSourceReader::open_folder()
{
    if ((m_src_dir = opendir (m_folder_path.c_str())) == NULL)
        THROW("ERROR: Failed opening the directory at " +m_folder_path);

    size_t read_counter = 0;
    while((m_entity = readdir (m_src_dir)) != NULL)
    {
        if(m_entity->d_type != DT_REG)
            continue;
            
        if(++read_counter <= _read_offset)
            continue;
        
        if((read_counter - _read_offset - 1)% _read_interval != 0)
            continue;
        //TODO: Add checking type, just files, not directories are needed

        std::string file_path = m_folder_path;
        file_path.append("/");
        file_path.append(m_entity->d_name);
        m_file_names.push_back(file_path);
    }
    if(m_file_names.size() > 0 )
        LOG("Total of "+TOSTR(m_file_names.size())+" images loaded from "+ m_folder_path )
    else
        THROW("Could not find any file in "+m_folder_path)

    m_curr_file_idx = 0;
    return Reader::Status::OK;
}

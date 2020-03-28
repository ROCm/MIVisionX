#pragma once
#include "commons.h"
#include <vector>
#include <condition_variable>
#include <CL/cl.h>
#include <queue>
#include "meta_data.h"
#include "device_manager.h"
#include "commons.h"

using MetaDataNamePair = std::pair<ImageNameBatch,pMetaDataBatch>;
class RingBuffer
{
public:
    explicit RingBuffer(unsigned buffer_depth);
    ~RingBuffer();
    size_t level();
    bool empty();
    ///\param mem_type
    ///\param dev
    ///\param sub_buffer_size
    ///\param sub_buffer_count
    void init(RaliMemType mem_type, DeviceResources dev, unsigned sub_buffer_size, unsigned sub_buffer_count);
    std::vector<void*> get_read_buffers() ;
    void* get_host_master_read_buffer();
    std::vector<void*> get_write_buffers();
    MetaDataNamePair& get_meta_data();
    void set_meta_data(ImageNameBatch names, pMetaDataBatch meta_data);
    void reset();
    void pop();
    void push();
    void unblock_reader();
    void unblock_writer();
    void release_all_blocked_calls();
    RaliMemType mem_type() { return _mem_type; }
    void block_if_empty();
    void block_if_full();
private:
    std::queue<MetaDataNamePair> _meta_ring_buffer;
    MetaDataNamePair _last_image_meta_data;
    void increment_read_ptr();
    void increment_write_ptr();
    bool full();
    const unsigned BUFF_DEPTH;
    unsigned _sub_buffer_size;
    unsigned _sub_buffer_count;
    std::mutex _lock;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::vector<std::vector<void*>> _dev_sub_buffer;
    std::vector<std::vector<unsigned char>> _host_master_buffers;
    std::vector<std::vector<void*>> _host_sub_buffers;
    bool _dont_block = false;
    RaliMemType _mem_type;
    DeviceResources _dev;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
    std::mutex  _names_buff_lock;
};

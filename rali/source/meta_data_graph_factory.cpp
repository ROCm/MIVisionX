
#include <memory>
#include "bounding_box_graph.h"
#include "meta_data_graph.h"
#include "meta_data_graph_factory.h"
#include "exception.h"


std::shared_ptr<MetaDataGraph> create_meta_data_graph(const MetaDataConfig& config) {
    switch(config.type()) {
        case MetaDataType::Label:
        {
            return nullptr;
        }
        case MetaDataType::BoundingBox:
        {
            return std::make_shared<BoundingBoxGraph>();
        }

        default:
            THROW("MetaDataReader type is unsupported");
    }
}

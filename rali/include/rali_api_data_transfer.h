#ifndef MIVISIONX_RALI_API_DATA_TRANSFER_H
#define MIVISIONX_RALI_API_DATA_TRANSFER_H
#include "rali_api_types.h"
/*! \brief
 *
*/
extern "C"  RaliStatus   RALI_API_CALL raliCopyToOutput(RaliContext context, unsigned char * out_ptr, size_t out_size);

/*! \brief
 *
*/
extern "C"  RaliStatus   RALI_API_CALL raliCopyToOutputFloat(RaliContext context, float * out_ptr, size_t out_size);

#endif //MIVISIONX_RALI_API_DATA_TRANSFER_H

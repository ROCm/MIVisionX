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
extern "C"  RaliStatus   RALI_API_CALL raliCopyToOutputTensor32(RaliContext rali_context, float *out_ptr,
                                                              RaliTensorLayout tensor_format, float multiplier0,
                                                              float multiplier1, float multiplier2, float offset0,
                                                              float offset1, float offset2,
                                                              bool reverse_channels);

extern "C"  RaliStatus   RALI_API_CALL raliCopyToOutputTensor16(RaliContext rali_context, half *out_ptr,
                                                              RaliTensorLayout tensor_format, float multiplier0,
                                                              float multiplier1, float multiplier2, float offset0,
                                                              float offset1, float offset2,
                                                              bool reverse_channels);
#endif //MIVISIONX_RALI_API_DATA_TRANSFER_H

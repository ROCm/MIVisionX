#ifndef MIVISIONX_RALI_API_PARAMETERS_H
#define MIVISIONX_RALI_API_PARAMETERS_H
#include "rali_api_types.h"

///
/// \param seed
extern "C"  void RALI_API_CALL raliSetSeed(long long unsigned seed);

///
/// \return
extern "C"  long long unsigned  RALI_API_CALL raliGetSeed();

///
/// \param start
/// \param end
/// \return
extern "C"  RaliIntParam  RALI_API_CALL raliCreateIntUniformRand(int start, int end);

///
/// \param start
/// \param end
/// \param input_obj
/// \return
extern "C"  RaliStatus RALI_API_CALL raliUpdateIntUniformRand(int start, int end, RaliIntParam updating_obj);

///
/// \param obj
/// \return
extern "C"  int RALI_API_CALL raliGetIntValue(RaliIntParam obj);

///
/// \param obj
/// \return
extern "C"  float RALI_API_CALL raliGetFloatValue(RaliFloatParam obj);

///
/// \param start
/// \param end
/// \return
extern "C"  RaliFloatParam  RALI_API_CALL raliCreateFloatUniformRand(float start, float end);

///
/// \param val
/// \return
extern "C"  RaliFloatParam  RALI_API_CALL raliCreateFloatParameter(float val);

///
/// \param val
/// \return
extern "C"  RaliIntParam  RALI_API_CALL raliCreateIntParameter(int val);

///
/// \param new_val
/// \param input_obj
/// \return
extern "C" RaliStatus  RALI_API_CALL raliUpdateFloatParameter(float new_val, RaliFloatParam input_obj);

///
/// \param new_val
/// \param input_obj
/// \return
extern "C" RaliStatus  RALI_API_CALL raliUpdateIntParameter(int new_val, RaliIntParam input_obj);

///
/// \param start
/// \param end
/// \param input_obj
/// \return
extern "C"  RaliStatus RALI_API_CALL raliUpdateFloatUniformRand(float start, float end, RaliFloatParam updating_obj);

///
/// \param values
/// \param frequencies
/// \param size
/// \return
extern "C"  RaliIntParam  RALI_API_CALL raliCreateIntRand(const int *values, const double *frequencies, unsigned size);

///
/// \param values
/// \param frequencies
/// \param size
/// \param updating_obj
/// \return
extern "C"  RaliStatus RALI_API_CALL raliUpdateIntRand(const int *values, const double *frequencies, unsigned size, RaliIntParam updating_obj);

/// Sets the parameters for a new or existing RaliFloatRandGen object
/// \param values
/// \param frequencies
/// \param size
/// \return
extern "C"  RaliFloatParam  RALI_API_CALL raliCreateFloatRand(const float *values, const double *frequencies, unsigned size);

///
/// \param values
/// \param frequencies
/// \param size
/// \param updating_obj
/// \return
extern "C"  RaliStatus RALI_API_CALL raliUpdateFloatRand(const float *values, const double *frequencies, unsigned size, RaliFloatParam updating_obj);

#endif //MIVISIONX_RALI_API_PARAMETERS_H

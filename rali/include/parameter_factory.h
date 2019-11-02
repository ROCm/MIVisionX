
#pragma once
#include <set>
#include <thread>
#include <mutex>
#include <memory>
#include "parameter_random.h"
#include "parameter_simple.h"


enum class RaliParameterType
{
    DETERMINISTIC = 0,
    RANDOM_UNIFORM,
    RANDOM_CUSTOM
};

struct IntParam
{ 
    IntParam(
            Parameter<int>* core,
            RaliParameterType type):
            core(core),
            type(type){}
    Parameter<int>* core;
    const RaliParameterType type;
};

struct FloatParam
{ 
    FloatParam(
            Parameter<float>* core,
            RaliParameterType type):
            core(core),
            type(type){}
    Parameter<float>* core;
    const RaliParameterType type;
};

Parameter<int>* core(IntParam* arg);

Parameter<float>* core(FloatParam* arg);

using pParam = std::variant<IntParam*,FloatParam*>;
using pParamCore = std::variant<Parameter<int>*, Parameter<float>*>;

bool validate_custom_rand_param(pParam arg);
bool validate_uniform_rand_param(pParam  rand_obj);
bool validate_simple_rand_param(pParam arg);

class ParameterFactory {
public:
    static ParameterFactory* instance();
    ~ParameterFactory();
    void renew_parameters();
    void set_seed(long long unsigned seed);
    unsigned long long get_seed();

    template<typename T>
    Parameter<T>* create_uniform_rand_param(T start, T end){
        auto gen = new UniformRand<T>(start, end);
        _parameters.insert(gen);
        return gen;
    }
    template<typename T>
    Parameter<T>* create_single_value_param(T value){
        auto gen = new SimpleParameter<T>(value);
        _parameters.insert(gen);
        return gen;
    }
    template<typename T>
    void destroy_param(Parameter<T>* param)
    {
        if(_parameters.find(param) != _parameters.end())
            _parameters.erase(param);
        delete param;
    }
    IntParam* create_uniform_int_rand_param(int start, int end);
    FloatParam* create_uniform_float_rand_param(float start, float end);
    IntParam* create_custom_int_rand_param(const int *value, const double *frequencies, size_t size);
    FloatParam* create_custom_float_rand_param(const float *value, const double *frequencies, size_t size);
    IntParam* create_single_value_int_param(int value);
    FloatParam* create_single_value_float_param(float value);
private:
    long long unsigned m_seed; 
    std::set<pParamCore> _parameters; //<! Keeps the random generators used to randomized the augmentation parameters
    static ParameterFactory* _instance;
    static std::mutex _mutex; 
    ParameterFactory();
};









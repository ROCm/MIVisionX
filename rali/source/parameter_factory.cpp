#include <cstdlib>
#include <ctime>
#include "parameter_factory.h"
#include "parameter_simple.h"
ParameterFactory* ParameterFactory::_instance = nullptr;
std::mutex ParameterFactory::_mutex;

bool validate_simple_rand_param(pParam arg)
{
    bool ret = true;
    std::visit(
            [&ret](auto&& arg)
            {
                if(arg == nullptr || arg->type != RaliParameterType::DETERMINISTIC)
                    ret = false;

            },
            arg);
    return ret;
}


bool validate_custom_rand_param(pParam arg)
{
    bool ret = true;
    std::visit(
        [&ret](auto&& arg)
        {
            if(arg == nullptr || arg->type != RaliParameterType::RANDOM_CUSTOM)
                ret = false;
            
        },
        arg);
    return ret;
}

bool validate_uniform_rand_param(pParam  rand_obj)
{
    bool ret = true;
    std::visit(
        [&ret](auto&& arg)
        {
            if(arg == nullptr || arg->type != RaliParameterType::RANDOM_UNIFORM)
                ret = false;
                           
        },
        rand_obj);
    return ret;
}

ParameterFactory::ParameterFactory()
{
    m_seed = 0;
    std::srand(m_seed);
}

ParameterFactory* ParameterFactory::instance() {
    
    if(_instance == nullptr)// For performance reasons
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if(_instance == nullptr) 
        {
            _instance = new ParameterFactory();
        }
    }
    return _instance;
}

ParameterFactory::~ParameterFactory() {
    for(auto&& rand_obj : _parameters)
            std::visit(
                [](auto&& arg)
                {
                    delete arg;
                },
                rand_obj);
}

void ParameterFactory::renew_parameters()
{
    for(auto&& rand_obj : _parameters)
        std::visit(
                [](auto&& arg)
                {
                    arg->renew();
                },
                rand_obj);
}

unsigned long long 
ParameterFactory::get_seed()
{   
    return m_seed; 
}

void 
ParameterFactory::set_seed(long long unsigned seed)
{ 
    m_seed = seed;
    std::srand(m_seed); 
}

IntParam* ParameterFactory::create_uniform_int_rand_param(int start, int end)
{
    auto gen = new UniformRand<int>(start, end);
    auto ret = new IntParam(gen, RaliParameterType::RANDOM_UNIFORM);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_uniform_float_rand_param(float start, float end)
{
    auto gen = new UniformRand<float>(start, end);
    auto ret = new FloatParam(gen, RaliParameterType::RANDOM_UNIFORM);
    _parameters.insert(gen);
    return ret;
}


IntParam* ParameterFactory::create_custom_int_rand_param(const int *value, const double *frequencies, size_t size)
{
    auto gen = new CustomRand<int>(value, frequencies, size);
    auto ret = new IntParam(gen, RaliParameterType::RANDOM_CUSTOM);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_custom_float_rand_param(const float *value, const double *frequencies, size_t size)
{
    auto gen = new CustomRand<float>(value, frequencies, size);
    auto ret = new FloatParam(gen, RaliParameterType::RANDOM_CUSTOM);
    _parameters.insert(gen);
    return ret;
}


IntParam* ParameterFactory::create_single_value_int_param(int value)
{
    auto gen = new SimpleParameter<int>(value);
    auto ret = new IntParam(gen, RaliParameterType::DETERMINISTIC);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_single_value_float_param(float value)
{
    auto gen = new SimpleParameter<float>(value);
    auto ret = new FloatParam(gen, RaliParameterType::DETERMINISTIC);
    _parameters.insert(gen);
    return ret;
}

Parameter<int>* core(IntParam* arg)
{
    if(!arg)
        return nullptr;
    return arg->core;
}

Parameter<float>* core(FloatParam* arg)
{
    if(!arg)
        return nullptr;
    return arg->core;
}

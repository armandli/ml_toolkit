#ifndef ML_COMMON
#define ML_COMMON

#include <ctime>
#include <random>

namespace ML {

std::default_random_engine& get_default_random_engine(){
  static std::default_random_engine eng(time(0));
  return eng;
}

} //ML

#endif //ML_COMMON

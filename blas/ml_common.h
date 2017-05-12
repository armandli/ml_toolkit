#ifndef ML_COMMON
#define ML_COMMON

#include <ctime>
#include <random>

#ifndef MTX_BLOCK_SZ
#define MTX_BLOCK_SZ 8
#define MTX_BLOCK_MASK 7
#define MTX_MUL_DOUBLE_UNIT 2
#endif

namespace ML {

std::default_random_engine& get_default_random_engine(){
  static std::default_random_engine eng(time(0));
  return eng;
}

size_t round_up(size_t v){
  if (v == 1) return v;
  return (v & ~MTX_BLOCK_MASK) + (v & MTX_BLOCK_MASK ? MTX_BLOCK_SZ : 0);
}

} //ML

#endif //ML_COMMON

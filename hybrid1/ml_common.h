#ifndef ML_COMMON
#define ML_COMMON

#include <cassert>
#include <ctime>
#include <random>

#define D2Idx(pt, ir, ic, rs, cs) \
  (pt)[(ir) * (cs) + (ic)]

#define MTX_BLOCK_RSZ   4UL
#define MTX_BLOCK_RMASK 3UL
#define MTX_BLOCK_CSZ   32UL
#define MTX_BLOCK_CMASK 31UL

#define CPU_THREAD_SIZE 8UL

#define CUDA_MAX_TSZ    1024UL
#define CUDA_SQ_TSZ     32UL
#define CUDA_SLICE_SZ   4UL
#define CUDA_LOG_BANKS  5UL

#define ALIGNMENT_WIDTH 32UL

namespace ML {

std::default_random_engine& get_default_random_engine(){
  static std::default_random_engine eng(time(0));
  return eng;
}

enum class RandomizationType : unsigned {
  Uniform,
  Gaussian,
};

constexpr size_t roundup_row(size_t v){
  return (v & ~MTX_BLOCK_RMASK) + (v & MTX_BLOCK_RMASK ? MTX_BLOCK_RSZ : 0UL);
}

constexpr size_t roundup_col(size_t v){
  return (v & ~MTX_BLOCK_CMASK) + (v & MTX_BLOCK_CMASK ? MTX_BLOCK_CSZ : 0UL);
}

constexpr size_t pow2denom(size_t v){
  return ((v ^ (v - 1UL)) + 1UL) >> 1UL;
}

template <typename T>
constexpr size_t cdiv(T a, T b){
  return (a + b - 1) / b;
}

struct BlkSz {
  size_t rpb;
  size_t cpb;
};

BlkSz get_gpu_block_size(size_t rowstride, size_t colstride){
  //TODO: this is a bad assumption over reduction functions (e.g. each reduction reduce down the size of column
  //by 128, at some point the reduced size is no longer divisible by 32, e.g. column size 6776832UL
  assert(colstride % MTX_BLOCK_CSZ == 0);
  assert(rowstride % MTX_BLOCK_RSZ == 0);

  size_t dr = pow2denom(rowstride);
  size_t dc = pow2denom(colstride);

  size_t order = std::max(dr * dc / CUDA_MAX_TSZ, 1UL);
  size_t erow = std::max(dr / MTX_BLOCK_CSZ, 1UL);
  order /= erow;
  dr /= erow;
  dc /= order;

  return (BlkSz){dr, dc};
}

constexpr size_t gpu_size_threshold = 8000000UL;

constexpr bool is_gpu_optimized(size_t byte_size){
  return byte_size >= gpu_size_threshold;
}

enum ExprEvalEnv : unsigned {
  CPUExecution,
  GPUExecution,
};

constexpr size_t MIN_GPU_SIZE = 1000000;

struct RegSize {
  size_t rs;
  size_t cs;
};
bool operator == (const RegSize& a, const RegSize& b){
  return a.rs == b.rs && a.cs == b.cs;
}
bool operator != (const RegSize& a, const RegSize& b){
  return !(a == b);
}
bool operator < (const RegSize& a, const RegSize& b){
  if (a.rs < b.rs)                      return true;
  else if (a.rs == b.rs && a.cs < b.cs) return true;
  else                                  return false;
}

ExprEvalEnv det_execution_env(RegSize maxSize){
  if (maxSize.rs * maxSize.cs >= MIN_GPU_SIZE) return GPUExecution;
  else                                         return CPUExecution;
}

} //ML

#endif//ML_COMMON

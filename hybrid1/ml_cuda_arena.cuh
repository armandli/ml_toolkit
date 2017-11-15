#ifndef ML_CUDA_ARENA
#define ML_CUDA_ARENA

#include <cassert>
#include <ml_cuda_common.cuh>

namespace ML {
namespace CUDA {

class CUDArena {
  double* mMem;
  size_t  mRegSize;
  size_t  mCount;
  size_t  mSize;
public:
  CUDArena(const CUDArena&) = delete;
  CUDArena& operator=(const CUDArena&) = delete;

  CUDArena(size_t sz, size_t n): mMem(nullptr), mRegSize(sz), mCount(n), mSize(sz * n) {
    CUDADBG(cudaMalloc(&mMem, sizeof(double) * mRegSize * mCount));
  }

  ~CUDArena(){
    if (mMem != nullptr)
      CUDADBG(cudaFree(mMem));
  }

  void reset(size_t regsize, size_t count){
    size_t expected = regsize * count;
    if (mSize < expected){
      if (mMem != nullptr)
        CUDADBG(cudaFree(mMem));
      CUDADBG(cudaMalloc(&mMem, sizeof(double) * expected));
      mSize = expected;
    }
    mRegSize = regsize;
    mCount = count;
  }

  size_t regSize() const { return mRegSize; }
  size_t size() const { return mCount; }

  double* reg(size_t idx){
    assert(mMem != nullptr);
    assert(idx < mCount);

    return mMem + mRegSize * idx;
  }
};

} //CUDA
} //ML

#endif//ML_CUDA_ARENA

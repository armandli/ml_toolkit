#ifndef ML_CUDA_ARENA
#define ML_CUDA_ARENA

#include <cassert>
#include <limits>
#include <unordered_map>

#include <ml_cuda_common.cuh>

namespace ML {

class Mtx;

namespace CUDA {

//TODO: CUDA arena should be more robust, able to hold onto memory region despite being used elsewhere
class CUDArena {
  double* mMem;
  size_t  mRegSize;
  size_t  mCount;
  size_t  mSize;
  unordered_map<const Mtx*, double*> mCache;
  unordered_map<double*, const Mtx*> mLookup;
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

  void reserve(size_t regsize, size_t count){
    size_t expected = regsize * count;
    if (mSize < expected){
      if (mMem != nullptr)
        CUDADBG(cudaFree(mMem));
      CUDADBG(cudaMalloc(&mMem, sizeof(double) * expected));
      mSize = expected;
      mCache.clear();
      mLookup.clear();
    } else if (mSize != expected){
      mCache.clear();
      mLookup.clear();
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

  double* get_cache(const Mtx& mtx){
    unordered_map<const Mtx*, double*>::iterator = mCache.find(&mtx);
    if (it == mCache.end()) return nullptr;
    else                    return (*it).second;
  }

  void register_cache(const Mtx& mtx, double* r){
    if (r == nullptr){
      double* curr = mCache[mtx];
      mCache.erase(&mtx);
      mLookup.erase(curr);
    } else {
      mCache[&mtx] = r;
      mLookup[r] = &mtx;
    }
  }

  const Mtx* get_cached_at(double* m){
    decltype(mFree)::iterator it = mFree.find(m);
    if (it == mFree.end()) return nullptr;
    else                   return (*it).second;
  }

  void free_cache(double* m){
    decltype(mFree)::iterator it = mFree.find(m);
    if (it == mFree.end()) return;

    mCache.erase((*it).second);
    mFree.erase(m);
  }
};

} //CUDA
} //ML

#endif//ML_CUDA_ARENA

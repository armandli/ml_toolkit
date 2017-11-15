#ifndef ML_ARENA
#define ML_ARENA

#include <cassert>

namespace ML {

class MemArena {
  double* mMem;
  size_t  mRegSize;
  size_t  mCount;
  size_t  mSize;
public:
  MemArena(const MemArena&) = delete;
  MemArena& operator=(MemArena&) = delete;

  MemArena(): mMem(nullptr), mRegSize(0), mCount(0), mSize(0) {}
  MemArena(size_t regsize, size_t count): mRegSize(regsize), mCount(count), mSize(regsize * count) {
    mMem = new double[mSize];
  }
  ~MemArena(){
    if (mMem != nullptr)
      delete[] mMem;
  }

  void reset(size_t regsize, size_t count){
    size_t expect = regsize * count;
    if (mSize < expect){
      if (mMem != nullptr)
        delete[] mMem;
      mMem = new double[expect];
      mSize = expect;
    }
    mRegSize = regsize;
    mCount = count;
  }

  size_t regSize() const { return mRegSize; }
  size_t size() const { return mSize; }

  double* reg(size_t idx){
    assert(mMem != nullptr);
    assert(idx < mCount);
    return mMem + mRegSize * idx;
  }
};

} //ML

#endif//ML_ARENA

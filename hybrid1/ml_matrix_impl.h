#ifndef ML_MATRIX_IMPL
#define ML_MATRIX_IMPL

#include <vector>

#include <ml_matrix.h>
#include <ml_ssa_eval.h>

namespace ML {

class MemArena;
namespace CUDA {
  class CUDAArena;
}

void Mtx::evaluate(MemArena& arena){
  if (mSSA.get() == nullptr) return;

  memvaluateSSA(mSSA, arena);
}

void Mtx::evaluate(CUDA::CUDArena&){
  //TODO
}

};

#endif//ML_MATRIX_IMPL

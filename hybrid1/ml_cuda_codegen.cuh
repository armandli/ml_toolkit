#ifndef ML_CUDA_CODEGEN
#define ML_CUDA_CODEGEN

#include <vector>

#include <ml_cuda_common.h>
#include <ml_ssa_decl.h>
#include <ml_codegen.h>
#include <ml_cuda_arena.h>

namespace ML {

size_t estimate_gpu_local_registers(SSA& ssa, const std::vector<LiveSet>& live){
  size_t res = 0;
  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    size_t count = live[i].livein.size() + 1;
    res = std::max(res, count);
  }
  return res;
}

}

#endif//ML_CUDA_CODEGEN

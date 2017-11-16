#ifndef ML_CUDA_COMMON
#define ML_CUDA_COMMON

#ifndef CUDA_FAIL_CONTINUE
#define CUDA_FAIL_CONTINUE false
#else
#undef CUDA_FAIL_CONTINUE
#define CUDA_FAIL_CONTINUE true
#endif//CUDA_FAIL_CONTINUE
#define CUDADBG(call) SPPL::gassert((call), __FILE__, __LINE__, CUDA_FAIL_CONTINUE)

namespace ML {
namespace CUDA {
namespace SPPL {

inline void gassert(cudaError_t code, const char *file, int line, bool cont){
  if (code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (not cont) exit(code);
  }
}

} //SPPL
} //CUDA
} //ML  

#endif//ML_CUDA_COMMON

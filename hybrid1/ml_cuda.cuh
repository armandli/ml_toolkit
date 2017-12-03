#ifndef ML_CUDA
#define ML_CUDA

#include <cstddef>
#include <ctime>
#include <ml_common.h>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include <ml_cuda_common.cuh>

#define CUDA_CF_OFFSET(n) \
  ((n) >> CUDA_LOG_BANKS)

//TODO: more meaningful cukernel function names that indicate if it's 1D or 2D, and sum could be row sum, entire matrix sum, or col sum, need to clarify
//TODO: temporary buffer estimators for algorithm requiring additional device memory
//TODO: correctly handle column size smaller than 32 -> could lead to gpu compute block size to go wrong when also using slice size
//TODO: with sub-matrix, we are forced to do all cuda operations in 2D operations, need each 1D operation a 2D alternative

namespace ML {
namespace CUDA {
namespace SPPL {

__device__ size_t cuda_cf_offset(size_t n){
  return (n >> CUDA_LOG_BANKS);
}

} //SPPL

// matrix initialization operation
__global__ void const_init_2d_cukernel_pd(double* dst, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t idcols = rowid * colstride + cols;
  const size_t idmax = (rowid + 1) * colstride;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  if (rowid >= rows){
    while (id < idmax){
      dst[id] = 0.;
      id += inc;
    }
    return;
  }

  while (id < idcols){
    dst[id] = v;
    id += inc;
  }

  while (id < idmax){
    dst[id] = 0.;
    id += inc;
  }
}

void const_init_2d_cuda_pd(double* dst, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  const_init_2d_cukernel_pd<<< blocks, tpb >>>(dst, v, rows, cols, colstride);
}

// matrix random initialization operation
__global__ void rnd_init_seed_1d_cukernel_pd(unsigned seed, curandStatePhilox4_32_10_t* states, size_t sz){
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < sz)
    curand_init(seed, id, 0, &states[id]);
}

void rnd_init_seed_1d_cuda_pd(curandStatePhilox4_32_10_t* states, size_t sz){
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blk(std::max(sz / CUDA_MAX_TSZ, 1UL));
  unsigned seed = std::time(NULL);
  rnd_init_seed_1d_cukernel_pd<<< blk, tpb >>>(seed, states, sz);
}

__global__ void rnd_uniform_init_2d_cukernel_pd(double* dst, curandStatePhilox4_32_10_t* states, double lb, double ub, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  const size_t is_cols_odd = cols & 0x1UL;
  const size_t idcols2 = rowid * colstride + (cols & ~0x1UL);
  const size_t idmax = (rowid + 1) * colstride;
  const size_t inc2 = blockDim.x * gridDim.x * 2;

  size_t id = rowid * colstride + (blockDim.x * blockIdx.x + threadIdx.x) * 2;
  double range = ub - lb;

  if (rowid >= rows){
    while (id < idmax){
      double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
      *dst2 = make_double2(0., 0.);
      id += inc2;
    }
    return;
  }

  while (id < idcols2){
    double2 v2 = curand_uniform2_double(&states[id]);
    v2 = make_double2(v2.x * range + lb, v2.y * range + lb);
    double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
    *dst2 = v2;
    id += inc2;
  }

  if (id == idcols2 && is_cols_odd){
    double v = curand_uniform_double(&states[id]);
    dst[id] = v * range + lb;
    dst[id + 1] = 0.;
    id += inc2;
  }

  while (id < idmax){
    double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
    *dst2 = make_double2(0., 0.);
    id += inc2;
  }
}

void rnd_uniform_init_2d_cuda_pd(double* dst, curandStatePhilox4_32_10_t* states, double lb, double ub, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, std::max(colstride / CUDA_SLICE_SZ, 32UL));
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / (CUDA_SLICE_SZ << 1UL), 1UL), rowstride / bs.rpb);
  rnd_uniform_init_2d_cukernel_pd<<< blocks, tpb >>>(dst, states, lb, ub, rows, cols, colstride);
}

__global__ void rnd_normal_init_2d_cukernel_pd(double* dst, curandStatePhilox4_32_10_t* states, double ex, double sd, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  const size_t is_cols_odd = cols & 0x1;
  const size_t idcols2 = rowid * colstride + (cols & ~0x1UL);
  const size_t idmax = (rowid + 1) * colstride;
  const size_t inc2 = blockDim.x * gridDim.x * 2;

  size_t id = rowid * colstride + (blockDim.x * blockIdx.x + threadIdx.x) * 2;

  if (rowid >= rows){
    while (id < idmax){
      double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
      *dst2 = make_double2(0., 0.);
      id += inc2;
    }
    return;
  }

  while (id < idcols2){
    double2 v2 = curand_normal2_double(&states[id]);
    v2 = make_double2(v2.x * sd + ex, v2.y * sd + ex);
    double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
    *dst2 = v2;
    id += inc2;
  }

  if (id == idcols2 && is_cols_odd){
    double v = curand_normal_double(&states[id]);
    dst[id] = v * sd + ex;
    dst[id + 1] = 0.;
    id += inc2;
  }

  while (id < idmax){
    double2* dst2 = reinterpret_cast<double2*>(&dst[id]);
    *dst2 = make_double2(0., 0.);
    id += inc2;
  }
}

void rnd_normal_init_2d_cuda_pd(double* dst, curandStatePhilox4_32_10_t* states, double ex, double sd, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, std::max(colstride / CUDA_SLICE_SZ, 32UL));
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / (CUDA_SLICE_SZ << 1UL), 1UL), rowstride / bs.rpb);
  rnd_normal_init_2d_cukernel_pd<<< blocks, tpb >>>(dst, states, ex, sd, rows, cols, colstride);
}

// matrix transpose operation
template <size_t BZ = CUDA_MAX_TSZ>
__global__ void transpose_2d_cukernel_pd(double* __restrict__ dst, const double* __restrict__ src){
  const size_t lcol = threadIdx.x;
  const size_t lrow = threadIdx.y;
  const size_t lcolsize = blockDim.x;
  const size_t lrowsize = blockDim.y;
  const size_t gcol = blockIdx.x;
  const size_t grow = blockIdx.y;
  const size_t colsize = lcolsize * gridDim.x;
  const size_t rowsize = lrowsize * gridDim.y;
  const size_t icol = lcolsize * gcol + lcol;
  const size_t irow = lrowsize * grow + lrow;
  const size_t dstlcol = lcol % lrowsize;
  const size_t dstlrow = lcol / lrowsize + lrow * (lcolsize / lrowsize);

  __shared__ double sm[BZ + CUDA_CF_OFFSET(BZ)];

  sm[lcol * lrowsize + lrow + SPPL::cuda_cf_offset(lcol * lrowsize + lrow)] = src[irow * colsize + icol];

  __syncthreads();

  dst[ gcol * lcolsize * rowsize + grow * lrowsize + dstlrow * rowsize + dstlcol ] = sm[lrow * lcolsize + lcol + SPPL::cuda_cf_offset(lrow * lcolsize + lcol)];
}

//TODO: check if this is going to work for boundary cases where memory size for source and destination are very different. e.g. transpose a matrix of 50 * 4
void transpose_2d_cuda_pd(double* __restrict__ dst, const double* __restrict__ src, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(colstride / bs.cpb, rowstride / bs.rpb);

  transpose_2d_cukernel_pd<><<< blocks, tpb >>>(dst, src);
}

// add operation
__global__ void add_1d_cukernel_pd(double* dst, const double* s1, const double* s2, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    dst[id] = s1[id] + s2[id];
    id += inc;
  }
}

void add_1d_cuda_pd(double* dst, const double* s1, const double* s2, size_t rowstride, size_t colstride){
  const size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  add_1d_cukernel_pd<<< blocks, tpb >>>(dst, s1, s2, sz);
}

__global__ void add_const_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = src[id] + v;
    id += inc;
  }
}

void add_const_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  add_const_2d_cukernel_pd<<< blocks, tpb >>>(dst, src, v, rows, cols, colstride);
}

// subtraction operation
__global__ void sub_1d_cukernel_pd(double* dst, const double* s1, const double* s2, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    dst[id] = s1[id] - s2[id];
    id += inc;
  }
}

void sub_1d_cuda_pd(double* dst, const double* s1, const double* s2, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  sub_1d_cukernel_pd<<< blocks, tpb >>>(dst, s1, s2, sz);
}

__global__ void sub_mc_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = src[id] - v;
    id += inc;
  }
}

void sub_mc_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  sub_mc_2d_cukernel_pd<<< blocks, tpb >>>(dst, src, v, rows, cols, colstride);
}

__global__ void sub_cm_2d_cukernel_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = v - src[id];
    id += inc;
  }
}

void sub_cm_2d_cuda_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  sub_cm_2d_cukernel_pd<<< blocks, tpb >>>(dst, v, src, rows, cols, colstride);
}

// element multiply operation
__global__ void emul_1d_cukernel_pd(double* dst, const double* s1, const double* s2, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    dst[id] = s1[id] * s2[id];
    id += inc;
  }
}

void emul_1d_cuda_pd(double* dst, const double* s1, const double* s2, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  emul_1d_cukernel_pd<<< blocks, tpb >>>(dst, s1, s2, sz);
}

__global__ void emul_const_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = src[id] * v;
    id += inc;
  }
}

void emul_const_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  emul_const_2d_cukernel_pd<<< blocks, tpb >>>(dst, src, v, rows, cols, colstride);
}

// element wise division
__global__ void ediv_2d_cukernel_pd(double* dst, const double* s1, const double* s2, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = s1[id] / s2[id];
    id += inc;
  }
}

void ediv_2d_cuda_pd(double* dst, const double* s1, const double* s2, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);

  ediv_2d_cukernel_pd<<< blocks, tpb >>>(dst, s1, s2, rows, cols, colstride);
}

__global__ void ediv_mc_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = src[id] / v;
    id += inc;
  }
}

void ediv_mc_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  ediv_mc_2d_cukernel_pd<<< blocks, tpb >>>(dst, src, v, rows, cols, colstride);
}

__global__ void ediv_cm_2d_cukernel_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc   = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = v / src[id];
    id += inc;
  }
}

void ediv_cm_2d_cuda_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  ediv_cm_2d_cukernel_pd<<< blocks, tpb >>>(dst, v, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void not_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const unsigned long long v = 0xFFFFFFFFFFFFFFFFUL;
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (src[id] == 0.) dst[id] = *(double*)(unsigned long long*)&v;
    else               dst[id] = 0.;
    id += inc;
  }
}

void not_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  not_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void isnan_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const unsigned long long v = 0xFFFFFFFFFFFFFFFFUL;
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (isnan(src[id])) dst[id] = *(double*)(unsigned long long*)&v;
    else                dst[id] = 0.;
    id += inc;
  }
}

void isnan_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  isnan_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void isnan0_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (isnan(src[id])) dst[id] = 0.;
    else                dst[id] = src[id];
    id += inc;
  }
}

void isnan0_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  isnan0_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void sqrt_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = sqrt(src[id]);
    id += inc;
  }
}

void sqrt_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  sqrt_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void gt_2d_cukernel_pd(double* dst, const double* s1, const double* s2, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = s1[id] > s2[id];
    id += inc;
  }
}

void gt_2d_cuda_pd(double* dst, const double* s1, const double* s2, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  gt_2d_cukernel_pd<<<blocks, tpb>>>(dst, s1, s2, rows, cols, colstride);
}


//TODO: unit test
__global__ void gt_mc_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = src[id] > v;
    id += inc;
  }
}

void gt_mc_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  gt_mc_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, v, rows, cols, colstride);
}

//TODO: unit test
__global__ void gt_cm_2d_cukernel_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = v > src[id];
    id += inc;
  }
}

void gt_cm_2d_cuda_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  gt_cm_2d_cukernel_pd<<<blocks, tpb>>>(dst, v, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void gt0_mc_2d_cukernel_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (src[id] > v) dst[id] = src[id];
    else             dst[id] = 0.;
    id += inc;
  }
}

void gt0_mc_2d_cuda_pd(double* dst, const double* src, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  gt0_mc_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, v, rows, cols, colstride);
}

//TODO: unit test
__global__ void gt0_cm_2d_cukernel_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (v > src[id]) dst[id] = src[id];
    else             dst[id] = 0.;
    id += inc;
  }
}

void gt0_cm_2d_cuda_pd(double* dst, double v, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  gt0_cm_2d_cukernel_pd<<<blocks, tpb>>>(dst, v, src, rows, cols, colstride);
}

//TODO: unit test
__global__ void mask_2d_cukernel_pd(double* dst, const double* src, const double* mask, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;
  
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    if (mask[id] == 0.) dst[id] = 0.;
    else                dst[id] = src[id];
    id += inc;
  }
}

void mask_2d_cuda_pd(double* dst, const double* src, const double* mask, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  mask_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, mask, rows, cols, colstride);
}

// max row coefficient operation
template <size_t blkcsz, size_t BZ = CUDA_MAX_TSZ>
__global__ void max_row_coeff_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t increment = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;
  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ double sm[BZ];

  sm[lid] = DBL_MIN;
  while (id < idmax){
    sm[lid] = fmax(sm[lid], src[id]);
    id += increment;
  }
  __syncthreads();

  if (blkcsz >= 1024){
    if (clid < 512) sm[lid] = fmax(sm[lid], sm[lid + 512]);
    __syncthreads();
  }
  if (blkcsz >= 512){
    if (clid < 256) sm[lid] = fmax(sm[lid], sm[lid + 256]);
    __syncthreads();
  }
  if (blkcsz >= 256){
    if (clid < 128) sm[lid] = fmax(sm[lid], sm[lid + 128]);
    __syncthreads();
  }
  if (blkcsz >= 128){
    if (clid < 64) sm[lid] = fmax(sm[lid], sm[lid + 64]);
    __syncthreads();
  }
  if (blkcsz >= 64){
    if (clid < 32) sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
  }
  if (clid < 16){
    if (blkcsz >= 32){
      sm[lid] = fmax(sm[lid], sm[lid + 16]);
      __syncthreads();
    }
    if (blkcsz >= 16){
      sm[lid] = fmax(sm[lid], sm[lid + 8]);
      __syncthreads();
    }
    if (blkcsz >= 8){
      sm[lid] = fmax(sm[lid], sm[lid + 4]);
      __syncthreads();
    }
    if (blkcsz >= 4){
      sm[lid] = fmax(sm[lid], sm[lid + 2]);
      __syncthreads();
    }
    if (blkcsz >= 2){
      sm[lid] = fmax(sm[lid], sm[lid + 1]);
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid];
}

void max_row_coeff_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  double* dstptr = tempbuf;
  const double* srcptr = src;

  size_t ecols = cols;
  size_t ecolstride = colstride;
  BlkSz bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
  size_t edstcols = std::max(ecolstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
  size_t edstcolstride = roundup_col(edstcols);

  while (edstcols > 1UL){
    dim3 tpb(bs.cpb, bs.rpb);
    dim3 blocks(edstcols, rowstride / bs.rpb);

    switch (bs.cpb){
      case 1024UL: max_row_coeff_2d_cukernel_pd<1024UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 512UL:  max_row_coeff_2d_cukernel_pd<512UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 256UL:  max_row_coeff_2d_cukernel_pd<256UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 128UL:  max_row_coeff_2d_cukernel_pd<128UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 64UL:   max_row_coeff_2d_cukernel_pd<64UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 32UL:   max_row_coeff_2d_cukernel_pd<32UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 16UL:   max_row_coeff_2d_cukernel_pd<16UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 8UL:    max_row_coeff_2d_cukernel_pd<8UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 4UL:    max_row_coeff_2d_cukernel_pd<4UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      default:     assert(!!!"Invalid column per block for max_row_coeff_2d_cuda_pd"); break;
    }

    srcptr = dstptr;
    dstptr = dstptr + edstcolstride * rowstride;
    ecols = edstcols;
    ecolstride = edstcolstride;
    bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
    edstcols = std::max(ecolstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
    edstcolstride = roundup_col(edstcols);
  }

  edstcolstride = 1UL;
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(edstcols, rowstride / bs.rpb);

  switch (bs.cpb){
    case 1024UL: max_row_coeff_2d_cukernel_pd<1024UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 512UL:  max_row_coeff_2d_cukernel_pd<512UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 256UL:  max_row_coeff_2d_cukernel_pd<256UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 128UL:  max_row_coeff_2d_cukernel_pd<128UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 64UL:   max_row_coeff_2d_cukernel_pd<64UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 32UL:   max_row_coeff_2d_cukernel_pd<32UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 16UL:   max_row_coeff_2d_cukernel_pd<16UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 8UL:    max_row_coeff_2d_cukernel_pd<8UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 4UL:    max_row_coeff_2d_cukernel_pd<4UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    default:     assert(!!!"Invalid column per block for max_row_coeff_2d_cuda_pd"); break;
  }
}

//max row coefficient index operation
template <size_t blkcsz, size_t BZ = CUDA_MAX_TSZ>
__global__ void max_row_coeff_idx_l1_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ size_t sm[BZ];

  sm[lid] = rowid * colstride;
  while (id < idmax){
    if (src[id] > src[sm[lid]])
      sm[lid] = id;
    id += inc;
  }
  __syncthreads();

  if (blkcsz >= 256){
    if (clid < 128 && src[(size_t)sm[lid + 128]] > src[sm[lid]])
      sm[lid] = sm[lid + 128];
    __syncthreads();
  }
  if (blkcsz >= 128){
    if (clid < 64 && src[sm[lid + 64]] > src[sm[lid]])
      sm[lid] = sm[lid + 64];
    __syncthreads();
  }
  if (blkcsz >= 64){
    if (clid < 32 && src[sm[lid + 32]] > src[sm[lid]])
      sm[lid] = sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (blkcsz >= 32){
      if (src[sm[lid + 16]] > src[sm[lid]])
        sm[lid] = sm[lid + 16];
      __syncthreads();
    }
    if (blkcsz >= 16){
      if (src[sm[lid + 8]] > src[sm[lid]])
        sm[lid] = sm[lid + 8];
      __syncthreads();
    }
    if (blkcsz >= 8){
      if (src[sm[lid + 4]] > src[sm[lid]])
        sm[lid] = sm[lid + 4];
      __syncthreads();
    }
    if (blkcsz >= 4){
      if (src[sm[lid + 2]] > src[sm[lid]])
        sm[lid] = sm[lid + 2];
      __syncthreads();
    }
    if (blkcsz >= 2){
      if (src[sm[lid + 1]] > src[sm[lid]])
        sm[lid] = sm[lid + 1];
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid] - rowid * colstride;
}

template <size_t blkcsz, size_t BZ = CUDA_MAX_TSZ>
__global__ void max_row_coeff_idx_l2_2d_cukernel_pd(double* dst, const double* src, const double* sidx, size_t rows, size_t cols, size_t colstride, size_t srccolstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ size_t sm[BZ];
  sm[lid] = rowid * srccolstride;
  while (id < idmax){
    if (src[(size_t)sidx[id] + rowid * srccolstride] > src[sm[lid]])
      sm[lid] = (size_t)sidx[id] + rowid * srccolstride;
    id += inc;
  }
  __syncthreads();

  if (blkcsz >= 256){
    if (clid < 128 && src[sm[lid + 128]] > src[sm[lid]])
        sm[lid] = sm[lid + 128];
    __syncthreads();
  }
  if (blkcsz >= 128){
    if (clid < 64 && src[sm[lid + 64]] > src[sm[lid]])
        sm[lid] = sm[lid + 64];
    __syncthreads();
  }
  if (blkcsz >= 64){
    if (clid < 32 && src[sm[lid + 32]] > src[sm[lid]])
        sm[lid] = sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (blkcsz >= 32){
      if (src[sm[lid + 16]] > src[sm[lid]])
        sm[lid] = sm[lid + 16];
      __syncthreads();
    }
    if (blkcsz >= 16){
      if (src[sm[lid + 8]] > src[sm[lid]])
        sm[lid] = sm[lid + 8];
      __syncthreads();
    }
    if (blkcsz >= 8){
      if (src[sm[lid + 4]] > src[sm[lid]])
        sm[lid] = sm[lid + 4];
      __syncthreads();
    }
    if (blkcsz >= 4){
      if (src[sm[lid + 2]] > src[sm[lid]])
        sm[lid] = sm[lid + 2];
      __syncthreads();
    }
    if (blkcsz >= 2){
      if (src[sm[lid + 1]] > src[sm[lid]])
        sm[lid] = sm[lid + 1];
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid] - rowid * srccolstride;
}

void max_row_coeff_idx_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  size_t ecols = cols;
  size_t ecolstride = colstride;
  BlkSz bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
  size_t edstcols = ecolstride / bs.cpb / CUDA_SLICE_SZ;
  size_t edstcolstride = edstcols == 1UL ? 1UL : roundup_col(edstcols);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blk(edstcols, rowstride / bs.rpb);

  double* dstptr = edstcols == 1UL ? dst : tempbuf;

  //level 1 max row coeff index
  switch (bs.cpb){
    case 256UL:  max_row_coeff_idx_l1_2d_cukernel_pd<256UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 128UL:  max_row_coeff_idx_l1_2d_cukernel_pd<128UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 64UL:   max_row_coeff_idx_l1_2d_cukernel_pd<64UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 32UL:   max_row_coeff_idx_l1_2d_cukernel_pd<32UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 16UL:   max_row_coeff_idx_l1_2d_cukernel_pd<16UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 8UL:    max_row_coeff_idx_l1_2d_cukernel_pd<8UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    case 4UL:    max_row_coeff_idx_l1_2d_cukernel_pd<4UL><<< blk, tpb >>>(dstptr, src, rows, cols, ecolstride, edstcolstride); break;
    default:     assert(!!!"Invalid column per block for max_row_coeff_idx_l1_2d_cuda_pd"); break;
  }

  while (dstptr != dst){
    double* srcptr = dstptr;
    ecols = edstcols;
    ecolstride = edstcolstride;
    bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
    edstcols = std::max(ecolstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
    edstcolstride = edstcols == 1UL ? 1UL : roundup_col(edstcols);
    dstptr = edstcols == 1UL ? dst : srcptr + ecolstride * rowstride;

    tpb = dim3(bs.cpb, bs.rpb);
    blk = dim3(edstcols, rowstride / bs.rpb);

    switch (bs.cpb){
      case 256UL: max_row_coeff_idx_l2_2d_cukernel_pd<256UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 128UL: max_row_coeff_idx_l2_2d_cukernel_pd<128UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 64UL: max_row_coeff_idx_l2_2d_cukernel_pd<64UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 32UL: max_row_coeff_idx_l2_2d_cukernel_pd<32UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 16UL: max_row_coeff_idx_l2_2d_cukernel_pd<16UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 8UL: max_row_coeff_idx_l2_2d_cukernel_pd<8UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      case 4UL: max_row_coeff_idx_l2_2d_cukernel_pd<8UL><<< blk, tpb >>>(dstptr, src, srcptr, rows, ecols, ecolstride, colstride, edstcolstride); break;
      default: assert(!!!"Invalid column per block for max_row_coeff_idx_l2_2d_cukernel_pd"); break;
    }
  }
}

// block column row summation operation
template <size_t blkcsz, size_t BZ = CUDA_MAX_TSZ>
__global__ void row_sum_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t increment = blockDim.x * gridDim.x;
  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;
  size_t idmax = rowid * colstride + cols;
  __shared__ double sm[BZ];

  sm[lid] = 0.;
  while (id < idmax){
    sm[lid] += src[id];
    id += increment;
  }
  __syncthreads();

  if (blkcsz >= 256){
    if (clid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (blkcsz >= 128){
    if (clid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (blkcsz >= 64){
    if (clid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (blkcsz >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (blkcsz >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (blkcsz >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (blkcsz >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (blkcsz >= 2){
      sm[lid] += sm[lid + 1];
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid];
}

void row_sum_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  double* dstptr = tempbuf;
  const double* srcptr = src;

  size_t ecols = cols;
  size_t ecolstride = colstride;
  BlkSz bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
  size_t edstcols = std::max(ecolstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
  size_t edstcolstride = roundup_col(edstcols);

  while (edstcols > 1UL){
    dim3 tpb(bs.cpb, bs.rpb);
    dim3 blocks(edstcols, rowstride / bs.rpb);

    switch (bs.cpb){
      case 256UL:  row_sum_2d_cukernel_pd<256UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 128UL:  row_sum_2d_cukernel_pd<128UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 64UL:   row_sum_2d_cukernel_pd<64UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 32UL:   row_sum_2d_cukernel_pd<32UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 16UL:   row_sum_2d_cukernel_pd<16UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 8UL:    row_sum_2d_cukernel_pd<8UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      case 4UL:    row_sum_2d_cukernel_pd<4UL><<< blocks, tpb >>>(dstptr, srcptr, rows, ecols, ecolstride, edstcolstride); break;
      default:     assert(!!!"Invalid column per block for row_sum_2d_cuda_pd"); break;
    }

    srcptr = dstptr;
    dstptr = dstptr + edstcolstride * rowstride;
    ecols = edstcols;
    ecolstride = edstcolstride;
    bs = get_gpu_block_size(rowstride, std::max(ecolstride / CUDA_SLICE_SZ, 32UL));
    edstcols = std::max(ecolstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
    edstcolstride = roundup_col(edstcols);
  }

  edstcolstride = 1UL;
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(edstcols, rowstride / bs.rpb);

  switch (bs.cpb){
    case 256UL:  row_sum_2d_cukernel_pd<256UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 128UL:  row_sum_2d_cukernel_pd<128UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 64UL:   row_sum_2d_cukernel_pd<64UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 32UL:   row_sum_2d_cukernel_pd<32UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 16UL:   row_sum_2d_cukernel_pd<16UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 8UL:    row_sum_2d_cukernel_pd<8UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    case 4UL:    row_sum_2d_cukernel_pd<4UL><<< blocks, tpb >>>(dst, srcptr, rows, ecols, ecolstride, edstcolstride); break;
    default:     assert(!!!"Invalid column per block for row_sum_2d_cuda_pd"); break;
  }
}

// mean block operation, depend on sum block operation
__global__ void row_mean_1d_cukernel_pd(double* dst, const double* src, size_t colcount, size_t cols){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  double dcolcount = (double)colcount;

  while (id < cols){
    dst[id] = src[id] / dcolcount;
    id += inc;
  }
}

void row_mean_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  row_sum_2d_cuda_pd(dst, src, rows, cols, rowstride, colstride, tempbuf);

  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(rowstride / CUDA_MAX_TSZ, 1UL));
  row_mean_1d_cukernel_pd<<< blocks, tpb >>>(dst, dst, cols, rows);
}

// sigmoid block operation
__global__ void sigmoid_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = 1. / (1. + exp(src[id] * -1.));
    id += inc;
  }
}

void sigmoid_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  sigmoid_2d_cukernel_pd<<< blocks, tpb >>>(dst, src, rows, cols, colstride);
}

//TODO: unit test
// deriviative of sigmoid operation
__global__ void dsigmoid_2d_cukernel_pd(double* dst, const double* dm, const double* m, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    double a = dm[id];
    double b = m[id];
    dst[id] = a * b * (1. - b);
    id += inc;
  }
}

void dsigmoid_2d_cuda_pd(double* dst, const double* dm, const double* m, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  dsigmoid_2d_cukernel_pd<<<blocks, tpb>>>(dst, dm, m, rows, cols, colstride);
}

// tanh block operation
__global__ void tanh_1d_cukernel_pd(double* dst, const double* src, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    dst[id] = tanh(src[id]);
    id += inc;
  }
}

void tanh_1d_cuda_pd(double* dst, const double* src, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  tanh_1d_cukernel_pd<<< blocks, tpb >>>(dst, src, sz);
}

//TODO: unit test
// deriviative of tanh operation
__global__ void dtanh_2d_cukernel_pd(double* dst, const double* dm, const double* m, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    double a = dm[id];
    double b = m[id];
    dst[id] = a * (1. - b * b);
    id += inc;
  }
}

void dtanh_2d_cuda_pd(double* dst, const double* dm, const double* m, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  dtanh_2d_cukernel_pd<<<blocks, tpb>>>(dst, dm, m, rows, cols, colstride);
}

// relu block operation
__global__ void relu_1d_cukernel_pd(double* dst, const double* src, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    dst[id] = fmax(src[id], 0.);
    id += inc;
  }
}

void relu_1d_cuda_pd(double* dst, const double* src, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  relu_1d_cukernel_pd<<< blocks, tpb >>>(dst, src, sz);
}

// drelu block operation
__global__ void drelu_1d_cukernel_pd(double* dst, const double* dm, const double* m, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    //TODO: not very efficient
    if (m[id] > 0.) dst[id] = dm[id];
    else            dst[id] = 0.;
    id += inc;
  }
}

void drelu_1d_cuda_pd(double* dst, const double* dm, const double* m, size_t rowstride, size_t colstride){
  size_t sz = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(sz / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  drelu_1d_cukernel_pd<<< blocks, tpb >>>(dst, dm, m, sz);
}

// L2 loss block operation
template <size_t BZ = CUDA_MAX_TSZ>
__global__ void square_sum_1d_cukernel_pd(double* dst, const double* src, size_t sz){
  const size_t lid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ double sm[BZ];

  sm[lid] = 0.;
  while (id < sz){
    double v = src[id];
    sm[lid] += v * v;
    id += inc;
  }
  __syncthreads();

  if (BZ >= 1024){
    if (lid < 512) sm[lid] += sm[lid + 512];
    __syncthreads();
  }
  if (BZ >= 512){
    if (lid < 256) sm[lid] += sm[lid + 256];
    __syncthreads();
  }
  if (BZ >= 256){
    if (lid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (BZ >= 128){
    if (lid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (BZ >= 64){
    if (lid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (lid < 16){
    if (BZ >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (BZ >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (BZ >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (BZ >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (BZ >= 2)
      sm[lid] += sm[lid + 1];
  }

  if (lid == 0) dst[bid] = sm[lid];
}

template <size_t BZ = CUDA_MAX_TSZ>
__global__ void sum_1d_cukernel_pd(double* dst, const double* src, size_t sz){
  const size_t lid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ double sm[BZ];

  sm[lid] = 0.;
  while (id < sz){
    sm[lid] += src[id];
    id += inc;
  }
  __syncthreads();

  if (BZ >= 1024){
    if (lid < 512) sm[lid] += sm[lid + 512];
    __syncthreads();
  }
  if (BZ >= 512){
    if (lid < 256) sm[lid] += sm[lid + 256];
    __syncthreads();
  }
  if (BZ >= 256){
    if (lid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (BZ >= 128){
    if (lid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (BZ >= 64){
    if (lid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (lid < 16){
    if (BZ >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (BZ >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (BZ >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (BZ >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (BZ >= 2)
      sm[lid] += sm[lid + 1];
  }

  if (lid == 0) dst[bid] = sm[lid];
}

//TODO: maybe not having to use the conditional switch between which kernel to call
void loss_l2_1d_cuda_pd(double* dst, double* src, double reg, size_t rowstride, size_t colstride, double* tempbuf){
  double* dstptr = tempbuf;
  const double* srcptr = src;

  bool is_first = true;

  size_t esize = rowstride * colstride;
  size_t edstsize = std::max(esize / CUDA_SLICE_SZ / CUDA_MAX_TSZ, 1UL);

  while (edstsize > 1UL){
    dim3 tpb(CUDA_MAX_TSZ);
    dim3 blocks(edstsize);

    if (is_first){
      is_first = false;
      square_sum_1d_cukernel_pd<><<< blocks, tpb >>>(dstptr, srcptr, esize);
    } else {
      sum_1d_cukernel_pd<><<< blocks, tpb >>>(dstptr, srcptr, esize);
    }

    srcptr = dstptr;
    dstptr = dstptr + roundup_row(edstsize);
    esize = edstsize;
    edstsize = std::max(esize / CUDA_SLICE_SZ / CUDA_MAX_TSZ, 1UL);
  }

  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(1);

  if (is_first)
    square_sum_1d_cukernel_pd<><<< blocks, tpb >>>(dst, srcptr, esize);
  else
    sum_1d_cukernel_pd<><<< blocks, tpb >>>(dst, srcptr, esize);

  dim3 final_tpb(1);
  dim3 final_blocks(1);
  emul_const_2d_cukernel_pd<<< final_blocks, final_tpb >>>(dst, dst, 0.5 * reg, 1, 1, 1);
}

//TODO: unit test
void loss_l2_2d_cuda_pd(double* dst, double* src, double reg, size_t rowstride, size_t colstride, double* tempbuf){
  //TODO
}

// row based softmax operation
__global__ void exp_2d_cukernel_pd(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  while (id < idmax){
    dst[id] = exp(src[id]);
    id += inc;
  }
}

void exp_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  BlkSz bs = get_gpu_block_size(rowstride, colstride);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);
  exp_2d_cukernel_pd<<<blocks, tpb>>>(dst, src, rows, cols, colstride);
}


template <size_t BZ>
__global__ void div_vec_2d_cukernel_pd(double* dst, const double* src, const double* vec, size_t rows, size_t cols, size_t colstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t clid = threadIdx.x;
  const size_t rlid = threadIdx.y;
  const size_t idmax = rowid * colstride + cols;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double sm[BZ];

  if (clid == 0){
    sm[rlid] = vec[rowid];
  }
  __syncthreads();

  while (id < idmax){
    dst[id] = src[id] / sm[rlid];
    //dst[id] = src[id] / vec[rowid];
    id += inc;
  }
}

void softmax_row_2d_cuda_pd(double* dst, const double* src, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: element-wise exponentiate
  BlkSz ebs = get_gpu_block_size(rowstride, std::max(colstride / CUDA_SLICE_SZ, 32UL));
  dim3 etpb(ebs.cpb, ebs.rpb);
  dim3 eblocks(std::max(colstride / ebs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / ebs.rpb);
  exp_2d_cukernel_pd<<< eblocks, etpb >>>(dst, src, rows, cols, colstride);

  //step2: row based summation
  double* rowsum = tempbuf;
  double* rowsumbuf = tempbuf + rowstride;
  row_sum_2d_cuda_pd(rowsum, dst, rows, cols, rowstride, colstride, rowsumbuf);

  //step3: element-wise division on the sum
  BlkSz bs = get_gpu_block_size(rowstride, std::max(colstride / CUDA_SLICE_SZ, 32UL));
  dim3 dtpb(bs.cpb, bs.rpb);
  dim3 dblocks(std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL), rowstride / bs.rpb);

  switch (bs.rpb){
    case 32UL: div_vec_2d_cukernel_pd<32UL><<< dblocks, dtpb >>>(dst, dst, rowsum, rows, cols, colstride); break;
    case 16UL: div_vec_2d_cukernel_pd<16UL><<< dblocks, dtpb >>>(dst, dst, rowsum, rows, cols, colstride); break;
    case 8UL: div_vec_2d_cukernel_pd<8UL><<< dblocks, dtpb >>>(dst, dst, rowsum, rows, cols, colstride); break;
    case 4UL: div_vec_2d_cukernel_pd<4UL><<< dblocks, dtpb >>>(dst, dst, rowsum, rows, cols, colstride); break;
    default: assert(!!!"Invalid row per block for softmax_row_2d_cuda_pd"); break;
  }
}

// MSE loss and accuracy computation operations
template <size_t BZ>
__global__ void diff_square_sum_1d_cukernel_pd(double* dst, const double* a, const double* b, size_t rsize){
  const size_t lid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double sm[BZ];

  sm[lid] = 0.;
  while (id < rsize){
    double v = a[id] - b[id];
    sm[lid] += v * v;
    id += inc;
  }
  __syncthreads();

  if (BZ >= 1024){
    if (lid < 512) sm[lid] += sm[lid + 512];
    __syncthreads();
  }
  if (BZ >= 512){
    if (lid < 256) sm[lid] += sm[lid + 256];
    __syncthreads();
  }
  if (BZ >= 256){
    if (lid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (BZ >= 128){
    if (lid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (BZ >= 64){
    if (lid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (lid < 16){
    if (BZ >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (BZ >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (BZ >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (BZ >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (BZ >= 2){
      sm[lid] += sm[lid + 1];
      __syncthreads();
    }
  }

  if (lid == 0) dst[bid] = sm[lid];
}

void mse_loss_1d_cuda_pd(double* dst, const double* o, const double* y, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: difference square sum of matrix, may not be able to sum up the entire matrix
  size_t esize = rowstride * colstride;
  size_t edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  dim3 dsstpb(CUDA_MAX_TSZ);
  dim3 dssblocks(edstsize);
  diff_square_sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< dssblocks, dsstpb >>>(tempbuf, o, y, esize);

  //step2: if matrix was not summed up in 1 layer, try multiple layers
  const double* srcptr = tempbuf;
  double* dstptr = tempbuf + roundup_row(edstsize);

  esize = edstsize;
  edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  while (edstsize > 1UL){
    dim3 tpb(CUDA_MAX_TSZ);
    dim3 blocks(edstsize);

    sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< blocks, tpb >>>(dstptr, srcptr, esize);

    srcptr = dstptr;
    dstptr = dstptr + roundup_row(edstsize);
    esize = edstsize;
    edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  }

  dim3 stpb(CUDA_MAX_TSZ);
  dim3 sblocks(1);
  sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< sblocks, stpb >>>(dst, srcptr, esize);

  //step3: divide the final value by the number of rows
  dim3 final_tpb(1);
  dim3 final_blocks(1);
  ediv_mc_2d_cukernel_pd<<< final_blocks, final_tpb >>>(dst, dst, (double)rows, 1, 1, 1); 
}

__global__ void sqrt_1d_cukernel_pd(double* dst, const double* src, size_t rsize){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < rsize){
    dst[id] = sqrt(src[id]);
    id += inc;
  }
}

void mse_accuracy_1d_cuda_pd(double* dst, const double* o, const double* y, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: difference square sum of matrix, may not be completely summed up into one matrix
  size_t esize = rowstride * colstride;
  size_t edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  dim3 dsstpb(CUDA_MAX_TSZ);
  dim3 dssblocks(edstsize);
  diff_square_sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< dssblocks, dsstpb >>>(tempbuf, o, y, esize);

  //step2: if matrix was not summed up in 1 layer, doing summation in additional layers
  const double* srcptr = tempbuf;
  double* dstptr = tempbuf + roundup_row(edstsize);

  esize = edstsize;
  edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  while (edstsize > 1UL){
    dim3 tpb(CUDA_MAX_TSZ);
    dim3 blocks(edstsize);

    sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< blocks, tpb >>>(dstptr, srcptr, esize);

    srcptr = dstptr;
    dstptr = dstptr + roundup_row(edstsize);
    esize = edstsize;
    edstsize = std::max(esize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL);
  }

  dim3 stpb(CUDA_MAX_TSZ);
  dim3 sblocks(1);
  sum_1d_cukernel_pd<CUDA_MAX_TSZ><<< sblocks, stpb >>>(dst, srcptr, esize);

  //step3: divide the final sum by number of rows * 2
  dim3 final_tpb(1);
  dim3 final_blocks(1);
  sqrt_1d_cukernel_pd<<< final_blocks, final_tpb >>>(dst, dst, 1);
  ediv_mc_2d_cukernel_pd<<< final_blocks, final_tpb >>>(dst, dst, (double)rows * 2., 1, 1, 1);
}

// MSVM loss and accraucy operation
template <size_t blkcsz, size_t RZ, size_t BZ = CUDA_MAX_TSZ>
__global__ void msvm_loss_2d_cukernel_pd(double* dst, const double* o, const double* yidx, double f, size_t rows, size_t cols, size_t colstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowid >= rows) return;

  const size_t rlid = threadIdx.y;
  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;
  
  __shared__ double sm[BZ];
  __shared__ double mm[RZ];

  //TODO: not the most efficient because of thread divergence
  if (clid == 0)
    mm[rlid] = o[rowid * colstride + (size_t)yidx[rowid]];
  __syncthreads();

  sm[lid] = 0.;
  while (id < idmax){
    double v = o[id] - mm[rlid] + f;
    v = fmax(v, 0.);
    sm[lid] += v * v;
    id += inc;
  }
  __syncthreads();

  if (blkcsz >= 256){
    if (clid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (blkcsz >= 128){
    if (clid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (blkcsz >= 64){
    if (clid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (blkcsz >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (blkcsz >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (blkcsz >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (blkcsz >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (blkcsz >= 2){
      sm[lid] += sm[lid + 1];
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid];
}

void msvm_loss_2d_cuda_pd(double* dst, const double* o, const double* y, double f, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: compute max coefficient index on Y
  double* ymaxcoeff = tempbuf;
  tempbuf = ymaxcoeff + rowstride;
  max_row_coeff_idx_2d_cuda_pd(ymaxcoeff, y, rows, cols, rowstride, colstride, tempbuf);

  //step2: MSVM reduce 1st level
  BlkSz bs = get_gpu_block_size(rowstride, std::max(colstride / CUDA_SLICE_SZ, 32UL));
  size_t edstcols = std::max(colstride / bs.cpb / CUDA_SLICE_SZ, 1UL);
  dim3 tpb(bs.cpb, bs.rpb);
  dim3 blk(edstcols, rowstride / bs.rpb);

  double* dstptr = tempbuf;

  switch (bs.cpb){
    case 256UL: msvm_loss_2d_cukernel_pd<256UL, 4UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 128UL: msvm_loss_2d_cukernel_pd<128UL, 8UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 64UL: msvm_loss_2d_cukernel_pd<64UL, 16UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 32UL: msvm_loss_2d_cukernel_pd<32UL, 32UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 16UL: msvm_loss_2d_cukernel_pd<16UL, 32UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 8UL: msvm_loss_2d_cukernel_pd<8UL, 32UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    case 4UL: msvm_loss_2d_cukernel_pd<4UL, 32UL><<< blk, tpb >>>(dstptr, o, ymaxcoeff, f, rows, cols, colstride, edstcols); break;
    default: assert(!!!"Invalid column per block for msvm_loss_2d_cukernel_pd"); break;
  }

  //step3: summation of all computed values of previous step in 1 dimension
  size_t fz = rows * edstcols;
  size_t fzstride = roundup_row(fz);
  while (fz > 1UL){
    size_t edstfz = cdiv(fzstride, CUDA_MAX_TSZ);
    double* srcptr = dstptr;
    dstptr = edstfz == 1UL ? dst : dstptr + fzstride;
    tpb = dim3(CUDA_MAX_TSZ);
    blk = dim3(edstfz);
    sum_1d_cukernel_pd<><<< blk, tpb >>>(dstptr, srcptr, fz);
    fz = edstfz;
    fzstride = roundup_row(fz);
  }

  //step4: final value minus f * f * rows, then divide by rows
  tpb = dim3(1);
  blk = dim3(1);
  double v = f * f * (double)rows;
  sub_mc_2d_cukernel_pd<<< blk, tpb >>>(dst, dst, v, 1, 1, 1);
  ediv_mc_2d_cukernel_pd<<< blk, tpb >>>(dst, dst, (double)rows, 1, 1, 1);
}

__global__ void compare_equal_1d_cukernel_pd(double* dst, double* a, double* b, size_t sz){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < sz){
    if (a[id] == b[id]) dst[id] = 1.;
    else                dst[id] = 0.;
    id += inc;
  }
}

// applies to MSVM as well
void accuracy_2d_cuda_pd(double* dst, const double* o, const double* y, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: compute max coefficient index on O and Y separately
  double* omaxcoeff = tempbuf;
  double* ymaxcoeff = omaxcoeff + rowstride;
  tempbuf = ymaxcoeff + rowstride;
  max_row_coeff_idx_2d_cuda_pd(omaxcoeff, o, rows, cols, rowstride, colstride, tempbuf);
  max_row_coeff_idx_2d_cuda_pd(ymaxcoeff, y, rows, cols, rowstride, colstride, tempbuf);

  //step2: compare resulting index values from O and Y, if match 1 otherwise 0
  double* comparison = tempbuf;
  tempbuf = comparison + rowstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blk(std::max(rowstride / CUDA_MAX_TSZ, 1UL));
  compare_equal_1d_cukernel_pd<<< blk, tpb >>>(comparison, omaxcoeff, ymaxcoeff, rows);

  //step3: summation of 1 and 0 computed previous step
  double* dstptr = comparison;
  size_t fz = rows;
  size_t fzstride = rowstride;
  while (fz > 1UL){
    size_t edstfz = cdiv(fzstride, CUDA_MAX_TSZ);
    double* srcptr = dstptr;
    dstptr = edstfz == 1UL ? dst : dstptr + fzstride;
    tpb = dim3(CUDA_MAX_TSZ);
    blk = dim3(edstfz);
    sum_1d_cukernel_pd<><<< blk, tpb >>>(dstptr, srcptr, fz);
    fz = edstfz;
    fzstride = roundup_row(fz);
  }

  //step4: divide the sum by rows
  tpb = dim3(1);
  blk = dim3(1);
  ediv_mc_2d_cukernel_pd<<< blk, tpb >>>(dst, dst, (double)rows, 1, 1, 1);
}

// CrossEntropy loss operation
template <size_t BZ = CUDA_MAX_TSZ>
__global__ void ce_loss_1d_cukernel_pd(double* dst, const double* o, const double* yidx, size_t rows, size_t colstride){
  const size_t lid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;

  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ double sm[BZ];

  sm[lid] = 0.;
  while (id < rows){
    sm[lid] += log(o[id * colstride + (size_t)yidx[id]]) * -1.;
    id += inc;
  }
  __syncthreads();

  if (BZ >= 1024){
    if (lid < 512) sm[lid] += sm[lid + 512];
    __syncthreads();
  }
  if (BZ >= 512){
    if (lid < 256) sm[lid] += sm[lid + 256];
    __syncthreads();
  }
  if (BZ >= 256){
    if (lid < 128) sm[lid] += sm[lid + 128];
    __syncthreads();
  }
  if (BZ >= 128){
    if (lid < 64) sm[lid] += sm[lid + 64];
    __syncthreads();
  }
  if (BZ >= 64){
    if (lid < 32) sm[lid] += sm[lid + 32];
    __syncthreads();
  }
  if (lid < 16){
    if (BZ >= 32){
      sm[lid] += sm[lid + 16];
      __syncthreads();
    }
    if (BZ >= 16){
      sm[lid] += sm[lid + 8];
      __syncthreads();
    }
    if (BZ >= 8){
      sm[lid] += sm[lid + 4];
      __syncthreads();
    }
    if (BZ >= 4){
      sm[lid] += sm[lid + 2];
      __syncthreads();
    }
    if (BZ >= 2)
      sm[lid] += sm[lid + 1];
  }

  if (lid == 0) dst[bid] = sm[lid];
}

void ce_loss_2d_cuda_pd(double* dst, const double* o, const double* y, size_t rows, size_t cols, size_t rowstride, size_t colstride, double* tempbuf){
  //step1: compute max coefficient index on Y
  double* ymaxcoeff = tempbuf;
  tempbuf = ymaxcoeff + rowstride;
  max_row_coeff_idx_2d_cuda_pd(ymaxcoeff, y, rows, cols, rowstride, colstride, tempbuf);

  //step2: cross entropy reduction 1st level
  double* dstptr = cdiv(rows, CUDA_MAX_TSZ) == 1UL ? dst : tempbuf;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blk(cdiv(rows, CUDA_MAX_TSZ));
  ce_loss_1d_cukernel_pd<><<< blk, tpb >>>(dstptr, o, ymaxcoeff, rows, colstride);

  //step3: summation of all computed values of previous step in 1 dimension
  size_t fz = cdiv(rows, CUDA_MAX_TSZ);
  size_t fzstride = roundup_row(fz);
  while (fz > 1UL){
    size_t edstfz = cdiv(fzstride, CUDA_MAX_TSZ);
    double* srcptr = dstptr;
    dstptr = edstfz == 1UL ? dst : dstptr + fzstride;
    blk = dim3(edstfz);
    sum_1d_cukernel_pd<><<< blk, tpb >>>(dstptr, srcptr, fz);
    fz = edstfz;
    fzstride = roundup_row(fz);
  }

  //step4: final value divide by rows
  tpb = dim3(1);
  blk = dim3(1);
  ediv_mc_2d_cukernel_pd<<< blk, tpb >>>(dst, dst, (double)rows, 1, 1, 1);
}

// row deriviative operation
__global__ void deriviative_row_1d_cukernel_pd(double* dst, const double* o, const double* y, double rows, size_t rsize){
  const size_t inc = blockDim.x * gridDim.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;

  while (id < rsize){
    double v = (o[id] - y[id]) / rows;
    if (isnan(v))
      v = 0.;
    dst[id] = v;
    id += inc;
  }
}

void deriviative_row_1d_cuda_pd(double* dst, const double* o, const double* y, size_t rows, size_t rowstride, size_t colstride){
  size_t rsize = rowstride * colstride;
  dim3 tpb(CUDA_MAX_TSZ);
  dim3 blocks(std::max(rsize / CUDA_MAX_TSZ / CUDA_SLICE_SZ, 1UL));
  deriviative_row_1d_cukernel_pd<<< blocks, tpb >>>(dst, o, y, (double)rows, rsize);
}

} //CUDA
} //ML

#endif//ML_CUDA

#include <mtx.h>
#include <ctime>
#include <float.h>
#include <algorithm>
#include <iostream>

using namespace std;

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ  = BSZ * TSZ;

template <size_t CZ, size_t RZ>
__global__ void max_mtx_coeffs_idx_r_l1(size_t* dst, const double* src, size_t cols, size_t colstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ size_t sm[CZ * RZ];
  sm[lid] = 0;
  while (id < idmax){
    if (src[id] > src[sm[lid]])
      sm[lid] = id;
    id += inc;
  }
  __syncthreads();

  if (CZ >= 256){
    if (clid < 128 && src[sm[lid + 128]] > src[sm[lid]])
        sm[lid] = sm[lid + 128];
    __syncthreads();
  }
  if (CZ >= 128){
    if (clid < 64 && src[sm[lid + 64]] > src[sm[lid]])
        sm[lid] = sm[lid + 64];
    __syncthreads();
  }
  if (CZ >= 64){
    if (clid < 32 && src[sm[lid + 32]] > src[sm[lid]])
        sm[lid] = sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (CZ >= 32){
      if (src[sm[lid + 16]] > src[sm[lid]])
        sm[lid] = sm[lid + 16];
      __syncthreads();
    }
    if (CZ >= 16){
      if (src[sm[lid + 8]] > src[sm[lid]])
        sm[lid] = sm[lid + 8];
      __syncthreads();
    }
    if (CZ >= 8){
      if (src[sm[lid + 4]] > src[sm[lid]])
        sm[lid] = sm[lid + 4];
      __syncthreads();
    }
    if (CZ >= 4){
      if (src[sm[lid + 2]] > src[sm[lid]])
        sm[lid] = sm[lid + 2];
      __syncthreads();
    }
    if (CZ >= 2){
      if (src[sm[lid + 1]] > src[sm[lid]])
        sm[lid] = sm[lid + 1];
      __syncthreads();
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid] - rowid * colstride;
}

template <size_t CZ, size_t RZ>
__global__ void max_mtx_coeffs_idx_r_l2(size_t* dst, const double* src, const size_t* sidx, size_t cols, size_t colstride, size_t srccolstride, size_t dstcolstride){
  const size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;

  const size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t clid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t inc = blockDim.x * gridDim.x;
  const size_t idmax = rowid * colstride + cols;

  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ size_t sm[CZ * RZ];
  sm[lid] = 0;
  while (id < idmax){
    if (src[sidx[id] + rowid * srccolstride] > src[sm[lid]])
      sm[lid] = sidx[id] + rowid * srccolstride;
    id += inc;
  }
  __syncthreads();

  if (CZ >= 256){
    if (clid < 128 && src[sm[lid + 128]] > src[sm[lid]])
        sm[lid] = sm[lid + 128];
    __syncthreads();
  }
  if (CZ >= 128){
    if (clid < 64 && src[sm[lid + 64]] > src[sm[lid]])
        sm[lid] = sm[lid + 64];
    __syncthreads();
  }
  if (CZ >= 64){
    if (clid < 32 && src[sm[lid + 32]] > src[sm[lid]])
        sm[lid] = sm[lid + 32];
    __syncthreads();
  }
  if (clid < 16){
    if (CZ >= 32){
      if (src[sm[lid + 16]] > src[sm[lid]])
        sm[lid] = sm[lid + 16];
      __syncthreads();
    }
    if (CZ >= 16){
      if (src[sm[lid + 8]] > src[sm[lid]])
        sm[lid] = sm[lid + 8];
      __syncthreads();
    }
    if (CZ >= 8){
      if (src[sm[lid + 4]] > src[sm[lid]])
        sm[lid] = sm[lid + 4];
      __syncthreads();
    }
    if (CZ >= 4){
      if (src[sm[lid + 2]] > src[sm[lid]])
        sm[lid] = sm[lid + 2];
      __syncthreads();
    }
    if (CZ >= 2){
      if (src[sm[lid + 1]] > src[sm[lid]])
        sm[lid] = sm[lid + 1];
      __syncthreads();
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid] - rowid * srccolstride;
}

void max_mtx_coeff_idx_r_cpu(size_t* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  for (size_t i = 0; i < rows; ++i){
    size_t maxidx = 0;
    for (size_t j = 0; j < cols; ++j)
      if (src[i * colstride + j] > src[i * colstride + maxidx])
        maxidx = j;
    dst[i] = maxidx;
  }
}

int main(){
  Mtx<double> a(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ);
  Mtx<size_t> b(false, 1, SZ), c(false, 1, SZ);
  Mtx<size_t> db(true, SZ, SZ), dc(true, 1, SZ);

  unary_init(a);

  max_mtx_coeff_idx_r_cpu(c.data, a.data, a.rows, a.cols, a.cols);

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);

  dim3 tpb1(256, 4);
  dim3 blocks1(SZ / 256 / 4, SZ / 4);
  max_mtx_coeffs_idx_r_l1<256, 4><<< blocks1, tpb1 >>>(db.data, da.data, SZ, SZ, SZ / 256 / 4);

  dim3 tpb2(32, 32);
  dim3 blocks2(1, SZ / 32);
  max_mtx_coeffs_idx_r_l2<32, 32><<< blocks2, tpb2 >>>(dc.data, da.data, db.data, 2, 2, 2048, 1);

  cudaMemcpy(b.data, dc.data, sizeof(double) * dc.rows * dc.cols, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < SZ; ++i)
    if (b.data[i] != c.data[i])
      cout << i << ": " << b.data[i] << " " << c.data[i] << endl;
}

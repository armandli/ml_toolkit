#include <mtx.h>
#include <ctime>
#include <float.h>
#include <x86intrin.h>
#include <algorithm>
#include <iostream>

using namespace std;

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ  = BSZ * TSZ;

template <size_t stride>
__global__ void max_mtx_coeffs_r(double* dst, const double* src, size_t cols, size_t colstride, size_t dstcolstride){
  size_t rowid = blockDim.y * blockIdx.y + threadIdx.y;
  size_t lid = blockDim.x * threadIdx.y + threadIdx.x;
  size_t clid = threadIdx.x;
  size_t bid = blockIdx.x;
  size_t id = rowid * colstride + blockDim.x * blockIdx.x + threadIdx.x;
  size_t idmax = rowid * colstride + cols;
  size_t increment = blockDim.x * gridDim.x;
  __shared__ double sm[1024];

  sm[lid] = DBL_MIN;
  while (id < idmax){
    sm[lid] = fmax(sm[lid], src[id]);
    id += increment;
  }
  __syncthreads();

  if (stride >= 256){
    if (clid < 128) sm[lid] = fmax(sm[lid], sm[lid + 128]);
    __syncthreads();
  }
  if (stride >= 128){
    if (clid < 64) sm[lid] = fmax(sm[lid], sm[lid + 64]);
    __syncthreads();
  }
  if (stride >= 64){
    if (clid < 32) sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
  }
  if (clid < 16){
    if (stride >= 32){
      sm[lid] = fmax(sm[lid], sm[lid + 16]);
      __syncthreads();
    }
    if (stride >= 16){
      sm[lid] = fmax(sm[lid], sm[lid + 8]);
      __syncthreads();
    }
    if (stride >= 8){
      sm[lid] = fmax(sm[lid], sm[lid + 4]);
      __syncthreads();
    }
    if (stride >= 4){
      sm[lid] = fmax(sm[lid], sm[lid + 2]);
      __syncthreads();
    }
    if (stride >= 2){
      sm[lid] = fmax(sm[lid], sm[lid + 1]);
      __syncthreads();
    }
  }

  if (clid == 0) dst[rowid * dstcolstride + bid] = sm[lid];
}

void max_mtx_coeffs_r_cuda(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  if (colstride >= 512){
    dim3 tpb(256, 4);
    dim3 blocks((colstride + 255) / 256 / 4, (rows + 3) / 4);
    max_mtx_coeffs_r<256><<< blocks, tpb >>>(dst, src, cols, colstride, std::max((colstride + 255) / 256 / 4, 32UL));
  } else if (colstride >= 256){
    dim3 tpb(128, 8);
    dim3 blocks((colstride + 127) / 128 / 4, (rows + 7) / 8);
    max_mtx_coeffs_r<128><<< blocks, tpb >>>(dst, src ,cols, colstride, std::max((colstride + 127) / 128 / 4, 32UL));
  } else if (colstride >= 128){
    dim3 tpb(64, 16);
    dim3 blocks((colstride + 63) / 64 / 4, (rows + 15) / 16);
    max_mtx_coeffs_r<64><<< blocks, tpb >>>(dst, src, cols, colstride, std::max((colstride + 63) / 64 / 4, 32UL));
  } else {
    dim3 tpb(32, 32);
    dim3 blocks(std::max((colstride + 31) / 32 / 4, 1UL), (rows + 31) / 32);
    max_mtx_coeffs_r<32><<< blocks, tpb >>>(dst, src, cols, colstride, std::max((colstride + 31) / 32 / 4, 1UL));
  }
}

double max_coeff_sse(const double* src, size_t rsize){
  __m256d maxv = _mm256_set1_pd(std::numeric_limits<double>::min());
  for (size_t ic = 0; ic < (rsize & ~4UL); ic += 4UL){
    __m256d a = _mm256_loadu_pd(&src[ic]);
    maxv = _mm256_max_pd(maxv, a);
  }
  double maxa[4];
  _mm256_storeu_pd(maxa, maxv);
  double mv = *std::max_element(maxa, maxa + 4UL);
  for (size_t ic = (rsize & ~4UL); ic < rsize; ++ic)
    mv = std::max(mv, src[ic]);
  return mv;
}

void max_coeffs_sse(double* dst, const double* src, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    const double* row = &src[ir * colstride];
    dst[ir] = max_coeff_sse(row, cols);
  }
}

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, 1, SZ), c(false, 1, SZ), d(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, 32), dd(true, SZ, 1);

  unary_init(a);

  clock_t start = clock();
  max_coeffs_sse(c.data, a.data, SZ, SZ, SZ);
  cout << "CPU Time: " << (clock() - start) << " us" << endl;

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);

  max_mtx_coeffs_r_cuda(db.data, da.data, SZ, SZ, SZ);
  max_mtx_coeffs_r_cuda(dd.data, db.data, SZ, SZ / 256 / 4, 32);

  cudaMemcpy(b.data, dd.data, sizeof(double) * dd.rows * dd.cols, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < SZ; ++i)
    if (c.data[i] != b.data[i])
      cout << "Row " << i << " Differ: " << b.data[i] << " " << c.data[i] << endl;
}

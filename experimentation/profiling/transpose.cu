#include <mtx.h>
#include <x86intrin.h>

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ = BSZ * TSZ;

//simple implementation
__global__ void transpose_kernel1(double* dst, const double* src){
  size_t colsize = blockDim.x * gridDim.x;
  size_t rowsize = blockDim.y * gridDim.y;
  size_t icol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t irow = blockDim.y * blockIdx.y + threadIdx.y;

  dst[icol * rowsize + irow] = src[irow * colsize + icol];
}

//use shared memory to induce coalescence
//~50% performance improvement
__global__ void transpose_kernel2(double* dst, const double* src){
  size_t colsize = blockDim.x * gridDim.x;
  size_t rowsize = blockDim.y * gridDim.y;
  size_t icol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t irow = blockDim.y * blockIdx.y + threadIdx.y;
  size_t lcol = threadIdx.x;
  size_t lrow = threadIdx.y;
  size_t lcolsize = blockDim.x;
  size_t lrowsize = blockDim.y;
  size_t gcol = blockIdx.x;
  size_t grow = blockIdx.y;

  __shared__ double cache[TSZ * TSZ];

  cache[lcol * lrowsize + lrow] = src[irow * colsize + icol];

  __syncthreads();

  dst[gcol * lcolsize * rowsize + grow * lrowsize + lrow * rowsize + lcol] = cache[lrow * lcolsize + lcol];
}

#define NUM_BANKS 32UL
#define LOG_NUM_BANKS 5UL

__device__ size_t CONFLICT_FREE_OFFSET(size_t n){
  return (n >> LOG_NUM_BANKS);
}

//optimization: avoiding bank conflict
__global__ void transpose_kernel3(double* dst, const double* src){
  size_t colsize = blockDim.x * gridDim.x;
  size_t rowsize = blockDim.y * gridDim.y;
  size_t icol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t irow = blockDim.y * blockIdx.y + threadIdx.y;
  size_t lcol = threadIdx.x;
  size_t lrow = threadIdx.y;
  size_t lcolsize = blockDim.x;
  size_t lrowsize = blockDim.y;
  size_t gcol = blockIdx.x;
  size_t grow = blockIdx.y;

  __shared__ double cache[TSZ * TSZ + 8];

  cache[lcol * lrowsize + lrow + CONFLICT_FREE_OFFSET(lcol * lrowsize + lrow)] = src[irow * colsize + icol];

  __syncthreads();

  dst[gcol * lcolsize * rowsize + grow * lrowsize + lrow * rowsize + lcol] = cache[lrow * lcolsize + lcol + CONFLICT_FREE_OFFSET(lrow * lcolsize + lcol)];
}

//increase block size and shared cache size
__global__ void transpose_kernel4(double* dst, const double* src){
  size_t colsize = blockDim.x * gridDim.x;
  size_t rowsize = blockDim.y * gridDim.y;
  size_t icol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t irow = blockDim.y * blockIdx.y + threadIdx.y;
  size_t lcol = threadIdx.x;
  size_t lrow = threadIdx.y;
  size_t lcolsize = blockDim.x;
  size_t lrowsize = blockDim.y;
  size_t gcol = blockIdx.x;
  size_t grow = blockIdx.y;

  __shared__ double cache[32 * 32 + 32];

  cache[lcol * lrowsize + lrow + CONFLICT_FREE_OFFSET(lcol * lrowsize + lrow)] = src[irow * colsize + icol];

  __syncthreads();

  dst[gcol * lcolsize * rowsize + grow * lrowsize + lrow * rowsize + lcol] = cache[lrow * lcolsize + lcol + CONFLICT_FREE_OFFSET(lrow * lcolsize + lcol)];
}

void transpose_block4x4(double* __restrict__ const dst, const double* __restrict__ const src, size_t rowstride, size_t colstride){
  for (size_t i = 0; i < rowstride; i += 4)
    for (size_t j = 0; j < colstride; j += 4){
      double* d = &dst[j * rowstride + i];
      const double* s = &src[i * colstride + j];

      __m256d r1 = _mm256_loadu_pd(&s[colstride * 0]);
      __m256d r2 = _mm256_loadu_pd(&s[colstride * 1]);
      __m256d r3 = _mm256_loadu_pd(&s[colstride * 2]);
      __m256d r4 = _mm256_loadu_pd(&s[colstride * 3]);
    
      __m256d t1 = _mm256_unpacklo_pd(r1, r2);
      __m256d t2 = _mm256_unpackhi_pd(r1, r2);
      __m256d t3 = _mm256_unpacklo_pd(r3, r4);
      __m256d t4 = _mm256_unpackhi_pd(r3, r4);
    
      __m256d p1 = _mm256_permute2f128_pd(t1, t3, 0x20);
      __m256d p2 = _mm256_permute2f128_pd(t2, t4, 0x20);
      __m256d p3 = _mm256_permute2f128_pd(t1, t3, 0x31);
      __m256d p4 = _mm256_permute2f128_pd(t2, t4, 0x31);
    
      _mm256_storeu_pd(&d[rowstride * 0], p1);
      _mm256_storeu_pd(&d[rowstride * 1], p2);
      _mm256_storeu_pd(&d[rowstride * 2], p3);
      _mm256_storeu_pd(&d[rowstride * 3], p4);
    }
}

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, SZ, SZ), c(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, SZ);

  unary_init(a);

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);

  //dim3 blocks(BSZ, BSZ);
  //dim3 tpb(TSZ, TSZ);
  //transpose_kernel1<<< blocks, tpb >>>(db.data, da.data);
  //transpose_kernel2<<< blocks, tpb >>>(db.data, da.data);
  //transpose_kernel3<<< blocks, tpb >>>(db.data, da.data);

  dim3 blocks(64, 64);
  dim3 tpb(32, 32);
  transpose_kernel4<<< blocks, tpb >>>(db.data, da.data);

  cudaMemcpy(b.data, db.data, sizeof(double) * db.rows * db.cols, cudaMemcpyDeviceToHost);

  clock_t start = clock();
  transpose_block4x4(c.data, a.data, SZ, SZ);
  cout << "CPU time: " << (clock() - start) << "us" << endl;

  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (b.data[i * SZ + j] != c.data[i * SZ + j]){
        cout << i << " " << j << ": " << c.data[i * SZ + j] << " " << b.data[i * SZ + j] << endl;
      }
}

#include <mtx.h>
#include <float.h>
#include <x86intrin.h>
#include <algorithm>

using namespace std;

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ = BSZ * TSZ;

__global__ void sum8(double* dst, const double* src, size_t n){
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  size_t increment = blockDim.x * 2 * gridDim.x;
  __shared__ double sm[512];

  sm[lid] = 0.;
  while (id < n){
    sm[lid] += src[id] + src[id + lsize];
    id += increment;
  }
  __syncthreads();

  if (lid < 256) sm[lid] += sm[lid + 256];
  __syncthreads();
  if (lid < 128) sm[lid] += sm[lid + 128];
  __syncthreads();
  if (lid < 64)  sm[lid] += sm[lid + 64];
  __syncthreads();

  if (lid < 32){
    sm[lid] += sm[lid + 32];
    __syncthreads();
    sm[lid] += sm[lid + 16];
    __syncthreads();
    sm[lid] += sm[lid + 8];
    __syncthreads();
    sm[lid] += sm[lid + 4];
    __syncthreads();
    sm[lid] += sm[lid + 2];
    __syncthreads();
    sm[lid] += sm[lid + 1];
  }

  if (lid == 0) dst[gid] = sm[0];
}

////only works on Kepler chips
//__device__ inline double __shfl_down_dbl(double var, unsigned lane, unsigned width = 32){
//  int2 a = *reinterpret_cast<int2*>(&var);
//  a.x = __shfl_down(a.x, lane, width);
//  a.y = __shfl_down(a.y, lane, width);
//  return *reinterpret_cast<double*>(&a);
//}
//
////optimization: use __shfl_down to avoid __syncthreads and do the last 32 lane reduce
//__global__ void sum9(double* dst, const double* src, size_t n){
//  size_t lid = threadIdx.x;
//  size_t lsize = blockDim.x;
//  size_t gid = blockIdx.x;
//  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
//  size_t increment = blockDim.x * 2 * gridDim.x;
//  __shared__ double sm[512];
//
//  sm[lid] = 0.;
//  while (id < n){
//    sm[lid] += src[id] + src[id + lsize];
//    id += increment;
//  }
//  __syncthreads();
//
//  if (lid < 256) sm[lid] += sm[lid + 256];
//  __syncthreads();
//  if (lid < 128) sm[lid] += sm[lid + 128];
//  __syncthreads();
//  if (lid < 64)  sm[lid] += sm[lid + 64];
//  __syncthreads();
//
//  if (lid < 32){
//    sm[lid] += sm[lid + 32];
//    __syncthreads();
//    double v = sm[lid + 16];
//    __shfl_down_dbl(v, 16);
//    v = sm[lid + 8];
//    __shfl_down_dbl(v, 8);
//    v = sm[lid + 4];
//    __shfl_down_dbl(v, 4);
//    v = sm[lid + 2];
//    __shfl_down_dbl(v, 2);
//    v = sm[lid + 1];
//    __shfl_down_dbl(v, 1);
//  }
//
//  if (lid == 0) dst[gid] = sm[0];
//}

double sum_sse(const double* src, size_t rsize){
  __m256d sumv = _mm256_set1_pd(0.);
  for (size_t i = 0; i < rsize; i += 4){
    __m256d a = _mm256_loadu_pd(&src[i]);
    sumv      = _mm256_add_pd(sumv, a);
  }
  double suma[4];
  _mm256_storeu_pd(suma, sumv);
  return std::accumulate(suma, suma + 4, 0.);
}

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, SZ), dc(true, 1, 1);

  unary_init(a);

  clock_t start = clock();
  double r = sum_sse(a.data, a.rows * a.cols);
  cout << "CPU Time: " << (clock() - start) << " us" << endl;

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);

  dim3 block1(2048);
  dim3 block2(1);
  dim3 tpb(512);
  sum8<<< block1, tpb >>>(db.data, da.data, 128 * 128 * 16 * 16);
  sum8<<< block2, tpb >>>(dc.data, db.data, 2048);

  //sum9<<< block1, tpb >>>(db.data, da.data, 128 * 128 * 16 * 16);
  //sum9<<< block2, tpb >>>(dc.data, db.data, 2048);

  cudaMemcpy(b.data, dc.data, sizeof(double) * dc.rows * dc.cols, cudaMemcpyDeviceToHost);

  cout << "CPU: " << r << " GPU: " << b.data[0] << endl;
}

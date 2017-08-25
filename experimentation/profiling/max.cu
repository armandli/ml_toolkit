#include <mtx.h>
#include <float.h>
#include <x86intrin.h>
#include <algorithm>

using namespace std;

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ = BSZ * TSZ;

//naive implementation using shared memory
__global__ void max_coeff1(double* dst, const double* src){
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = src[id];
  __syncthreads();

  for (size_t i = 2; i <= lsize; i *= 2){
    if (lid % i == 0) sm[lid] = fmax(sm[lid], sm[lid + (i >> 1)]);
    __syncthreads();
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: less divergent thread warp, 2000 times performance improvement
__global__ void max_coeff2(double* dst, const double* src){
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = src[id];
  __syncthreads();

  for (size_t i = 2; i <= lsize; i *= 2){
    //each warp has fewer chance of doing both positive and negative branch
    size_t idx1 = lid * i;
    size_t idx2 = idx1 + (i >> 1);
    if (idx2 < lsize) sm[idx1] = fmax(sm[idx1], sm[idx2]);
    __syncthreads();
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: optimize away bank conflict in shared memory, 40% performance improvement
__global__ void max_coeff3(double* dst, const double* src){
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = src[id];
  __syncthreads();

  //previous for loop has a 4-way bank conflict, this avoids it
  for (int i = lsize >> 1; i > 0; i >>= 1){
    if (lid < i) sm[lid] = fmax(sm[lid], sm[lid + i]);
    __syncthreads();
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: half the threads were used only for loading data from global memory. 100% performance improvement
__global__ void max_coeff4(double* dst, const double* src){
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = fmax(src[id], src[id + lsize]);
  __syncthreads();

  for (int i = lsize >> 1; i > 0; i >>= 1){
    if (lid < i) sm[lid] = fmax(sm[lid], sm[lid + i]);
    __syncthreads();
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: loop unrolling, no need to syncthreads when number of active threads is within a warp
//problem: it didn't work as expected
__global__ void max_coeff5(double* dst, const double* src){
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = fmax(src[id], src[id + lsize]);
  __syncthreads();

  for (size_t i = lsize >> 1; i > 32; i >>= 1){
    if (lid < i) sm[lid] = fmax(sm[lid], sm[lid + i]);
    __syncthreads();
  }

  if (lid < 32){
    sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 16]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 8]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 4]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 2]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 1]);
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: complete loop unroll
//assumption: max block size 256
__global__ void max_coeff6(double* dst, const double* src){
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  __shared__ double sm[TSZ * TSZ];

  sm[lid] = fmax(src[id], src[id + lsize]);
  __syncthreads();

  if (lid < 128) sm[lid] = fmax(sm[lid], sm[lid + 128]);
  __syncthreads();
  if (lid < 64)  sm[lid] = fmax(sm[lid], sm[lid + 64]);
  __syncthreads();

  if (lid < 32){
    sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 16]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 8]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 4]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 2]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 1]);
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: increase block size to 512
//result: performance degraded back to 300us
__global__ void max_coeff7(double* dst, const double* src){
  size_t lid = threadIdx.x;
  size_t lsize = blockDim.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  __shared__ double sm[512];

  sm[lid] = fmax(src[id], src[id + lsize]);
  __syncthreads();

  if (lid < 256) sm[lid] = fmax(sm[lid], sm[lid + 256]);
  __syncthreads();
  if (lid < 128) sm[lid] = fmax(sm[lid], sm[lid + 128]);
  __syncthreads();
  if (lid < 64)  sm[lid] = fmax(sm[lid], sm[lid + 64]);
  __syncthreads();

  if (lid < 32){
    sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 16]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 8]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 4]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 2]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 1]);
  }

  if (lid == 0) dst[gid] = sm[0];
}

//optimization: increase read bandwidth, algorithm cascading
//performance improved by 18%
__global__ void max_coeff8(double* dst, const double* src, size_t n){
  size_t lid = threadIdx.x;
  size_t gid = blockIdx.x;
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t increment = blockDim.x * gridDim.x;
  __shared__ double sm[512];

  sm[lid] = DBL_MIN;
  while (id < n){
    sm[lid] = fmax(sm[lid], src[id]);
    id += increment;
  }
  __syncthreads();

  if (lid < 256) sm[lid] = fmax(sm[lid], sm[lid + 256]);
  __syncthreads();
  if (lid < 128) sm[lid] = fmax(sm[lid], sm[lid + 128]);
  __syncthreads();
  if (lid < 64)  sm[lid] = fmax(sm[lid], sm[lid + 64]);
  __syncthreads();

  if (lid < 32){
    sm[lid] = fmax(sm[lid], sm[lid + 32]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 16]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 8]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 4]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 2]);
    __syncthreads();
    sm[lid] = fmax(sm[lid], sm[lid + 1]);
  }

  if (lid == 0) dst[gid] = sm[0];
}

//finally using float instead of double will also improvement performance by 100us

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

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, SZ), dc(true, SZ, SZ), dd(true, 1, 1);

  unary_init(a);

  //clock_t start  = clock();
  //double r = max_coeff_sse(a.data, SZ * SZ);
  //cout << "CPU time: " << (clock() - start) << " us" << endl;

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);

//  dim3 tpb(TSZ * TSZ);

  //layered reduction, each layer can only deal with local size 128
//  dim3 blocks1(BSZ * BSZ);
//  dim3 blocks2(BSZ * BSZ / (TSZ * TSZ));
//  dim3 blocks3(1);

//  max_coeff1<<< blocks1, tpb >>>(db.data, da.data);
//  max_coeff1<<< blocks2, tpb >>>(dc.data, db.data);
//  max_coeff1<<< blocks3, tpb >>>(dd.data, dc.data);

//  max_coeff2<<< blocks1, tpb >>>(db.data, da.data);
//  max_coeff2<<< blocks2, tpb >>>(dc.data, db.data);
//  max_coeff2<<< blocks3, tpb >>>(dd.data, dc.data);

//  max_coeff3<<< blocks1, tpb >>>(db.data, da.data);
//  max_coeff3<<< blocks2, tpb >>>(dc.data, db.data);
//  max_coeff3<<< blocks3, tpb >>>(dd.data, dc.data);

//  dim3 o1block1(BSZ * BSZ / 2);
//  dim3 o1block2(BSZ * BSZ / (TSZ * TSZ) / 2);
//  dim3 o1block3(1);

//  max_coeff4<<< o1block1, tpb >>>(db.data, da.data);
//  max_coeff4<<< o1block2, tpb >>>(dc.data, db.data);
//  max_coeff4<<< o1block3, tpb >>>(dd.data, dc.data);

//  max_coeff5<<< o1block1, tpb >>>(db.data, da.data);
//  max_coeff5<<< o1block2, tpb >>>(dc.data, db.data);
//  max_coeff5<<< o1block3, tpb >>>(dd.data, dc.data);

//  max_coeff6<<< o1block1, tpb >>>(db.data, da.data);
//  max_coeff6<<< o1block2, tpb >>>(dc.data, db.data);
//  max_coeff6<<< o1block3, tpb >>>(dd.data, dc.data);

  dim3 tpb2(512);

//  dim3 o2block1(4096);
//  dim3 o2block2(4);
//  dim3 o2block3(1);

//  max_coeff7<<< o2block1, tpb2 >>>(db.data, da.data);
//  max_coeff7<<< o2block2, tpb2 >>>(dc.data, db.data);
//  max_coeff7<<< o2block3, tpb2 >>>(dd.data, dc.data);

  dim3 o3block1(2048);
  dim3 o3block2(1);
  max_coeff8<<< o3block1, tpb2 >>>(db.data, da.data, 128 * 128 * 16 * 16);
  max_coeff8<<< o3block2, tpb2 >>>(dd.data, db.data, 2048);

  cudaMemcpy(b.data, dd.data, sizeof(double) * dd.rows * dd.cols, cudaMemcpyDeviceToHost);

  //cout << "CPU: " << r << " GPU: " << b.data[0] << endl;

  cout << "GPU: " << b.data[0] << endl;
}

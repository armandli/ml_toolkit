#include <x86intrin.h>
#include <mtx.h>

const size_t SZ = 2048;

typedef float ET;

// work inefficient prefix scan, 7.6160 us
template <size_t BZ>
__global__ void scan1(ET* dst, const ET* src, size_t n){
  const size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t lid = threadIdx.x;
  __shared__ double sm[BZ * 2];

  size_t pin = 1, pout = 0;
  //prefix scan, shifting values right by 1, first element should be 0.
  sm[pout * n + lid] = (lid > 0) ? src[id - 1] : 0.;
  __syncthreads();

  for (size_t offset = 1; offset < n; offset <<= 1){
    //buffer switch
    pin = 1 - pin;
    pout = 1 - pout;

    if (lid >= offset) sm[pout * n + lid] = sm[pin * n + lid] + sm[pin * n + lid - offset];
    else               sm[pout * n + lid] = sm[pin * n + lid];
    __syncthreads();
  }
  dst[id] = sm[pout * n + lid]; 
}

// work efficient implementation, can double the size of prefix scan block range to double the previous,
// performance 12 us
template <size_t BZ>
__global__ void scan2(ET* dst, const ET* src, size_t n){
  __shared__ ET sm[BZ * 2];
  const size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t lid = threadIdx.x;

  size_t offset = 1;
  sm[lid * 2] = src[2 * id];
  sm[lid * 2 + 1] = src[2 * id + 1];

  // sum up stage
  for (size_t d = n >> 1UL; d > 0; d >>= 1UL){
    __syncthreads();

    if (lid < d){
      size_t ai = offset * (2 * lid + 1) - 1;
      size_t bi = offset * (2 * lid + 2) - 1;
      sm[bi] += sm[ai];
    }

    offset <<= 1;
  }

  //clear the last eleemnt into 0.
  if (lid == 0) sm[n - 1] = 0.;

  // sum down stage
  for (size_t d = 1; d < n; d <<= 1UL){
    offset >>= 1;
    __syncthreads();

    if (lid < d){
      size_t ai = offset * (2 * lid + 1) - 1;
      size_t bi = offset * (2 * lid + 2) - 1;

      ET t = sm[ai];
      sm[ai] = sm[bi];
      sm[bi] += t;
    }
  }
  __syncthreads();

  dst[id * 2] = sm[lid * 2];
  dst[id * 2 + 1] = sm[lid * 2 + 1];
}

#define NBANK 32
#define LNBANK 5
#define BANK_OFFSET(n) \
  ((n) >> LNBANK)

//#define BANK_OFFSET(n) \
//  (((n) >> NBANK) + (n) >> (2 * LNBANK))

//improve performance by avoiding bank conflict, using a bank offset, does not see significant improvement
template <size_t BZ>
__global__ void scan3(ET* dst, const ET* src, size_t n){
  __shared__ ET sm[BZ * 2 + BANK_OFFSET(BZ * 2)];
  const size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t lid = threadIdx.x;

  size_t offset = 1;
  sm[lid * 2 + BANK_OFFSET(lid * 2)] = src[2 * id];
  sm[lid * 2 + 1 + BANK_OFFSET(lid * 2 + 1)] = src[2 * id + 1];

  // sum up stage
  for (size_t d = n >> 1UL; d > 0; d >>= 1UL){
    __syncthreads();

    if (lid < d){
      size_t ai = offset * (2 * lid + 1) - 1;
      size_t bi = offset * (2 * lid + 2) - 1;
      sm[bi + BANK_OFFSET(bi)] += sm[ai + BANK_OFFSET(ai)];
    }

    offset <<= 1;
  }

  //clear the last eleemnt into 0.
  if (lid == 0) sm[n - 1 + BANK_OFFSET(n - 1)] = 0.;

  // sum down stage
  for (size_t d = 1; d < n; d <<= 1UL){
    offset >>= 1;
    __syncthreads();

    if (lid < d){
      size_t ai = offset * (2 * lid + 1) - 1;
      size_t bi = offset * (2 * lid + 2) - 1;

      ET t = sm[ai + BANK_OFFSET(ai)];
      sm[ai + BANK_OFFSET(ai)] = sm[bi + BANK_OFFSET(bi)];
      sm[bi + BANK_OFFSET(bi)] += t;
    }
  }
  __syncthreads();

  dst[id * 2] = sm[lid * 2 + BANK_OFFSET(lid * 2)];
  dst[id * 2 + 1] = sm[lid * 2 + 1 + BANK_OFFSET(lid * 2 + 1)];
}

void scan_cpu(ET* dst, const ET* src, size_t n){
  dst[0] = 0.;
  for (size_t i = 1; i < n; ++i){
    dst[i] = dst[i-1] + src[i-1];
  }
}

int main(){
  Mtx<ET> a(false, 1, SZ), b(false, 1, SZ), c(false, 1, SZ);
  Mtx<ET> da(true, 1, SZ), db(true, 1, SZ);

  unary_init(a);

  cudaMemcpy(da.data, a.data, sizeof(ET) * a.rows * a.cols, cudaMemcpyHostToDevice);

  dim3 tpb(SZ / 2);
  dim3 blocks(1);
  scan2<SZ><<< blocks, tpb >>>(db.data, da.data, SZ);

  cudaMemcpy(b.data, db.data, sizeof(ET) * b.rows * b.cols, cudaMemcpyDeviceToHost);

  scan_cpu(c.data, a.data, SZ);

  for (size_t i = 0; i < SZ; ++i)
    if (b.data[i] != c.data[i]){
      cout << i << ": " << c.data[i] << " " << b.data[i] << endl;
    }
}

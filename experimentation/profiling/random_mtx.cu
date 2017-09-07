#include <mtx.h>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ  = BSZ * TSZ;

__global__ void random_init(unsigned seed, curandStatePhilox4_32_10_t* states){
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(seed, id, 0, &states[id]);
}

__global__ void random_normal(double* dst, curandStatePhilox4_32_10_t* states){
  unsigned id = blockDim.x * blockIdx.x  + threadIdx.x;
  unsigned idx = id * 2;
  double2* dst2 = reinterpret_cast<double2*>(&dst[idx]);
  *dst2 = curand_uniform2_double(&states[id]);
}

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, SZ);
  curandStatePhilox4_32_10_t* states;
  gpu_errchk(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * SZ * SZ / 2));

  dim3 blocks(BSZ * BSZ / 2);
  dim3 tpb(TSZ * TSZ);
  random_init<<< blocks, tpb >>>(time(NULL), states);

  random_normal<<< blocks, tpb >>>(da.data, states);
  random_normal<<< blocks, tpb >>>(db.data, states);

  cudaMemcpy(a.data, da.data, sizeof(double) * SZ * SZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data, db.data, sizeof(double) * SZ * SZ, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i <SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (a.data[i * SZ + j] == b.data[i * SZ + j]){
        cout << "Equal random value at same position " << (i * SZ + j) << endl;
      }

  cout << a << endl;
  cout << b << endl;
}

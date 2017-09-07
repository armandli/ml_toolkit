#include <x86intrin.h>
#include <mtx.h>

const size_t BSZ = 128;
const size_t TSZ = 16;
const size_t SZ  = BSZ * TSZ;

//naive implementation of matrix add
__global__ void add_block1(double* dst, const double* s1, const double* s2){
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  dst[gid] = s1[gid] + s2[gid];
}

//using vector types. no significant improvement
__global__ void add_block2(double* dst, const double* s1, const double* s2){
  size_t gid = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
  double2* s1d2 = (double2*)&s1[gid];
  double2* s2d2 = (double2*)&s2[gid];
  double2* dd2  = (double2*)&dst[gid];
  *dd2 = make_double2((*s1d2).x + (*s2d2).x, (*s1d2).y + (*s2d2).y);
}

__global__ void add_block3(double* dst, const double* s1, const double* s2, size_t sz){
  size_t id = blockDim.x *blockIdx.x + threadIdx.x;
  size_t increment = blockDim.x * gridDim.x;

  while (id < sz){
    dst[id] = s1[id] + s2[id];
    id += increment;
  }
}

void add_block_sse(double* dst, const double* s1, const double* s2, size_t rowstride, size_t colstride){
  for (size_t ir = 0; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += 4){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_add_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

int main(){
  Mtx<double> a(false, SZ, SZ), b(false, SZ, SZ), c(false, SZ, SZ), d(false, SZ, SZ);
  Mtx<double> da(true, SZ, SZ), db(true, SZ, SZ), dc(true, SZ, SZ);

  binary_init(a, b);

  cudaMemcpy(da.data, a.data, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);
  cudaMemcpy(db.data, b.data, sizeof(double) * b.rows * b.cols, cudaMemcpyHostToDevice);

//  dim3 blocks(BSZ * BSZ);
//  dim3 tpb(TSZ * TSZ);
//  add_block1<<< blocks, tpb >>>(dc.data, da.data, db.data);

//  dim3 blocks(BSZ * BSZ / 4);
//  dim3 tpb(TSZ * TSZ);
//  add_block2<<< blocks, tpb >>>(dc.data, da.data, db.data);

//  dim3 tpb2(1024);
//  dim3 blocks2(4096 / 2);
//  add_block2<<< blocks2, tpb2 >>>(dc.data, da.data, db.data);

  dim3 tpb3(256);
  dim3 blocks3(BSZ * BSZ * TSZ * TSZ / 256 / 4);
  add_block3<<< blocks3, tpb3 >>>(dc.data, da.data, db.data, BSZ * BSZ * TSZ * TSZ);

  cudaMemcpy(c.data, dc.data, sizeof(double) * c.rows * c.cols, cudaMemcpyDeviceToHost);

  add_block_sse(d.data, a.data, b.data, SZ, SZ);

  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (c.data[i * SZ + j] != d.data[i * SZ + j]){
        cout << i << " " << j << ": " << c.data[i * SZ + j] << " " << d.data[i * SZ + j] << endl;
      }
}

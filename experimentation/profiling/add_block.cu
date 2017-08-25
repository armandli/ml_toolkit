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
  size_t gid = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
  double2* s1d2 = (double2*)&s1[gid];
  double2* s2d2 = (double2*)&s2[gid];
  double2* dd2  = (double2*)&dst[gid];
  *dd2 = make_double2((*s1d2).x + (*s2d2).x, (*s1d2).y + (*s2d2).y);
  dd2[1] = make_double2(s1d2[1].x + s2d2[1].x, s1d2[1].y + s2d2[1].y);
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

  dim3 blocks(BSZ * BSZ);
  dim3 tpb(TSZ * TSZ);
  add_block1<<< blocks, tpb >>>(dc.data, da.data, db.data);

//  dim3 blocks(BSZ * BSZ / 4);
//  dim3 tpb(TSZ * TSZ);
//  add_block2<<< blocks, tpb >>>(dc.data, da.data, db.data);

  cudaMemcpy(c.data, dc.data, sizeof(double) * c.rows * c.cols, cudaMemcpyDeviceToHost);

  add_block_sse(d.data, a.data, b.data, SZ, SZ);

  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (c.data[i * SZ + j] != d.data[i * SZ + j]){
        cout << i << " " << j << ": " << c.data[i * SZ + j] << " " << d.data[i * SZ + j] << endl;
      }
}

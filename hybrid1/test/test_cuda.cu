#include <gtest/gtest.h>

#include <ml_common.h>
#include <ml_cuda.cuh>

using namespace std;
using namespace ML::CUDA;

typedef double* doubleptr;

struct InitTest : testing::Test {
  InitTest(){}
  ~InitTest(){
    CUDADBG(cudaFree(devblock));
    delete[] memblock;
    delete[] memblockref;
  }

  doubleptr devblock, memblock, memblockref;
  size_t rsz, csz;
};

TEST_F(InitTest, InitTest1){
  rsz = 64; csz = 64;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblock, 13., rsz, csz, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    for (size_t j = 0; j < csz; ++j)
      memblockref[i * csz + j] = 13.;

  EXPECT_TRUE(0 == memcmp(memblock, memblockref, sizeof(double) * rsz * csz));
}

TEST_F(InitTest, InitTest2){
  rsz = 96; csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblock, 14., rsz - 3, csz - 29, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_DOUBLE_EQ(0., memblock[csz - 2]);
  EXPECT_DOUBLE_EQ(0., memblock[(rsz - 2) * csz + csz - 32]);
  EXPECT_DOUBLE_EQ(0., memblock[(rsz - 2) * csz + csz - 1]);
}

struct RandInitTest : testing::Test {
  RandInitTest(){}
  ~RandInitTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devstates));
    delete[] memblock;
  }

  bool uniform_init_check(double* v, double l, double h, size_t r, size_t c, size_t rowstride, size_t colstride){
    size_t count = 0;
    for (size_t ir = 0; ir < r; ++ir){
      for (size_t ic = 0; ic < c; ++ic)
        if (v[ir * colstride + ic] < l or v[ir * colstride + ic] > h){
          if (v[ir * colstride + ic] < l)
            cout << "too low by " << (l - v[ir * colstride + ic]) << endl;
          if (v[ir * colstride + ic] > h)
            cout << "too high by " << (v[ir * colstride + ic] - h) << endl;
          count++;
        }
      for (size_t ic = c; ic < colstride; ++ic)
        if (v[ir * colstride + ic] != 0.)
          return false;
    }
    for (size_t ir = r; ir < rowstride; ++ir)
      for (size_t ic = 0; ic < colstride; ++ic)
        if (v[ir * colstride + ic] != 0.)
          return false;
    if (count > 0){
      return false;
    }
    return true;
  }

  bool normal_init_check(double* v, double ex, double sd, size_t r, size_t c, size_t rowstride, size_t colstride){
    size_t count = 0;
    for (size_t ir = 0; ir < r; ++ir){
      for (size_t ic = 0; ic < c; ++ic)
        if (v[ir * colstride + ic] < ex - 5 * sd or v[ir * colstride + ic] > ex + 5 * sd){
          cout << "out of 5 sd: " << v[ir * colstride + ic] << endl;
          count++;
        }
      for (size_t ic = c; ic < colstride; ++ic)
        if (v[ir * colstride + ic] != 0.)
          return false;
    }
    for (size_t ir = r; ir < rowstride; ++ir)
      for (size_t ic = 0; ic < colstride; ++ic)
        if (v[ir * colstride + ic] != 0.)
          return false;
    if (count > 0){
      return false;
    }
    return true;
  }

  curandStatePhilox4_32_10_t* devstates;
  doubleptr devblock, memblock;
  size_t rsz, csz;
};

TEST_F(RandInitTest, Uniform1){
  rsz = 32, csz = 64;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_uniform_init_2d_cuda_pd(devblock, devstates, 1., 2., rsz, csz, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(uniform_init_check(memblock, 1., 2., rsz, csz, rsz, csz));
}

TEST_F(RandInitTest, Uniform2){
  rsz = 4, csz = 1024;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_uniform_init_2d_cuda_pd(devblock, devstates, -1., 2., rsz, csz, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(uniform_init_check(memblock, -1., 2., rsz, csz, rsz, csz));
}

TEST_F(RandInitTest, Uniform3){
  rsz = 4, csz = 1024;
  size_t r = rsz - 3, c = csz - 3;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_uniform_init_2d_cuda_pd(devblock, devstates, 0., 10., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  
  EXPECT_TRUE(uniform_init_check(memblock, 0., 10., r, c, rsz, csz));
}

TEST_F(RandInitTest, Uniform4){
  rsz = 32, csz = 2048;
  size_t r = rsz - 3, c = csz - 3;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_uniform_init_2d_cuda_pd(devblock, devstates, -10., 10., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(uniform_init_check(memblock, -10., 10., r, c, rsz, csz));
}

TEST_F(RandInitTest, Uniform5){
  rsz = 32, csz = 2048;
  size_t r = rsz - 2, c = csz - 30;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_uniform_init_2d_cuda_pd(devblock, devstates, 50., 60., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(uniform_init_check(memblock, 50., 60., r, c, rsz, csz));
}


TEST_F(RandInitTest, Norm1){
  rsz = 32, csz = 64;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_normal_init_2d_cuda_pd(devblock, devstates, 1., 2., rsz, csz, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(normal_init_check(memblock, 1., 2., rsz, csz, rsz, csz));
}

TEST_F(RandInitTest, Norm2){
  rsz = 4, csz = 1024;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_normal_init_2d_cuda_pd(devblock, devstates, -1., 2., rsz, csz, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(normal_init_check(memblock, -1., 2., rsz, csz, rsz, csz));
}

TEST_F(RandInitTest, Norm3){
  rsz = 4, csz = 1024;
  size_t r = rsz - 3, c = csz - 3;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_normal_init_2d_cuda_pd(devblock, devstates, 0., 10., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  
  EXPECT_TRUE(normal_init_check(memblock, 0., 10., r, c, rsz, csz));
}

TEST_F(RandInitTest, Norm4){
  rsz = 32, csz = 2048;
  size_t r = rsz - 3, c = csz - 3;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_normal_init_2d_cuda_pd(devblock, devstates, 50., 1., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(normal_init_check(memblock, 50., 1., r, c, rsz, csz));
}

TEST_F(RandInitTest, Norm5){
  rsz = 32, csz = 2048;
  size_t r = rsz - 1, c = csz - 20;
  CUDADBG(cudaMalloc(&devstates, sizeof(curandStatePhilox4_32_10_t) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];

  rnd_init_seed_1d_cuda_pd(devstates, rsz * csz);
  rnd_normal_init_2d_cuda_pd(devblock, devstates, -10., 1., r, c, rsz, csz);
  CUDADBG(cudaMemcpy(memblock, devblock, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(normal_init_check(memblock, -10., 1., r, c, rsz, csz));
}


struct TransposeTest : testing::Test {
  void init_block(double* b, size_t r, size_t c){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        b[i * c + j] = i * c + j;
  }
  void transpose_cpu(double* __restrict__ dst, const double* __restrict__ src, size_t r, size_t c){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[j * r + i] = src[i * c + j];
  }

  TransposeTest(){}
  ~TransposeTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));

    delete[] memblock;
    delete[] memblockref;
    delete[] memblockdst;
  }

  doubleptr memblock, devblock, memblockref, devblockdst, memblockdst;
  size_t rsz, csz;
};

TEST_F(TransposeTest, TransposeTest1){
  rsz = 32, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest2){
  rsz = 96, csz = 96;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest3){
  rsz = 4, csz = 256;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest4){
  rsz = 16, csz = 512;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest5){
  rsz = 16, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest6){
  rsz = 8, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest7){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest8){
  rsz = 32, csz = 96;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(TransposeTest, TransposeTest9){
  rsz = 96, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_block(memblock, rsz, csz);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, rsz, csz);

  transpose_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

struct TransposeTest2 : testing::Test {
  void init_block(double* b, size_t r, size_t c, size_t cs){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        b[i * cs + j] = i * c + j;
  }
  void transpose_cpu(double* __restrict__ dst, const double* __restrict__ src, size_t r, size_t c, size_t scs, size_t dcs){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[j * dcs + i] = src[i * scs + j];
  }

  TransposeTest2(){}
  ~TransposeTest2(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));

    delete[] memblock;
    delete[] memblockref;
    delete[] memblockdst;
  }

  doubleptr memblock, devblock, memblockref, devblockdst, memblockdst;
  size_t srs, scs, drs, dcs;
};

TEST_F(TransposeTest2, TransposeTest1){
  srs = 50, scs = 4, drs = 4, dcs = 50;
  size_t srs_stride = ML::roundup_row(srs), scs_stride = ML::roundup_col(scs),
         drs_stride = ML::roundup_row(ML::roundup_col(drs)), dcs_stride = ML::roundup_col(dcs);

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * srs_stride * scs_stride));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * drs_stride * dcs_stride));

  memblock = new double[srs_stride * drs_stride];
  memblockdst = new double[drs_stride * dcs_stride];
  memblockref = new double[drs_stride * dcs_stride];
  init_block(memblock, srs, scs, scs_stride);
  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * srs_stride * scs_stride, cudaMemcpyHostToDevice));
  transpose_cpu(memblockref, memblock, srs, scs, scs_stride, dcs_stride);

  transpose_2d_cuda_pd(devblockdst, devblock, srs_stride, scs_stride, dcs_stride);
  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * drs_stride * dcs_stride, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < drs; ++ir)
    for (size_t ic = 0; ic < dcs; ++ic)
      EXPECT_DOUBLE_EQ(memblockref[ir * dcs_stride + ic], memblockdst[ir * dcs_stride + ic]);
}

struct BinaryMtxOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0.;
  }
  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    init_matrix(m, rowsize * colsize);

    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(-100., 100.);

    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void add_matrix(double* dst, double* a, double* b, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      dst[i] = a[i] + b[i];
  }

  void sub_matrix(double* dst, double* a, double* b, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      dst[i] = a[i] - b[i];
  }

  void emul_matrix(double* dst, double* a, double* b, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      dst[i] = a[i] * b[i];
  }

  void ediv_matrix(double* dst, double* a, double* b, size_t r, size_t c, size_t colsize){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colsize + j] = a[i * colsize + j] / b[i * colsize + j];
  }

  void drelu(double* dst, double* dm, double* m, size_t r, size_t c, size_t colsize){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colsize + j] = m[i * colsize + j]  > 0. ? dm[i * colsize + j] : 0.;
  }

  void deriviative(double* dst, double* o, double* y, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j){
        double v = (o[i * colstride + j] - y[i * colstride + j]) / (double)r;
        if (std::isnan(v)) v = 0.;
        dst[i * colstride + j] = v;
      }
  }

  BinaryMtxOpTest(){}
  ~BinaryMtxOpTest(){
    CUDADBG(cudaFree(devblock1));
    CUDADBG(cudaFree(devblock2));
    CUDADBG(cudaFree(devblockdst));

    delete[] memblock1;
    delete[] memblock2;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock1, devblock2, devblockdst, memblock1, memblock2, memblockref, memblockdst;
  size_t rsz, csz;
};

TEST_F(BinaryMtxOpTest, Add1){
  rsz = 64, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  add_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  add_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Add2){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  add_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  add_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Add3){
  rsz = 8, csz = 96;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz - 3, csz - 20, rsz, csz);
  init_matrix(memblock2, rsz - 3, csz - 20, rsz, csz);

  add_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  add_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}


TEST_F(BinaryMtxOpTest, Sub1){
  rsz = 64, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  sub_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Sub2){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  sub_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Sub3){
  rsz = 16, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz - 1, csz - 2, rsz, csz);
  init_matrix(memblock2, rsz - 1, csz - 2, rsz, csz);

  sub_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}


TEST_F(BinaryMtxOpTest, EMul1){
  rsz = 64, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  emul_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  emul_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, EMul2){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  emul_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  emul_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, EMul3){
  rsz = 16, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz - 2, csz - 16, rsz, csz);
  init_matrix(memblock2, rsz - 2, csz - 16, rsz, csz);

  emul_matrix(memblockref, memblock1, memblock2, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  emul_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}


TEST_F(BinaryMtxOpTest, EDiv1){
  rsz = 64, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  ediv_matrix(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, EDiv2){
  rsz = 32, csz = 96;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblockdst, rsz * csz);
  init_matrix(memblock1, rsz - 1, csz - 30, rsz, csz);
  init_matrix(memblock2, rsz - 1, csz - 30, rsz, csz);

  ediv_matrix(memblockref, memblock1, memblock2, rsz - 1, csz - 30, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz - 1, csz - 30, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Drelu1){
  rsz = 128, csz = 256;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  drelu(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  drelu_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, Drelu2){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  drelu(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  drelu_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, DeriviativeRow1){
  rsz = 64, csz = 64;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  deriviative(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  deriviative_row_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, DeriviativeRow2){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  deriviative(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  deriviative_row_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryMtxOpTest, DeriviativeRow3){
  rsz = 4, csz = 8192;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  deriviative(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  deriviative_row_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}


struct BinaryConstMtxOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0;
  }
  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < rowsize; ++i)
      for (size_t j = 0; j < colsize; ++j)
        m[i * colsize + j] = i * colsize + j;
  }

  void add_matrix(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = src[i * colstride + j] + v;
  }

  void sub_matrix_const(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = src[i * colstride + j] - v;
  }

  void sub_const_matrix(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = v - src[i * colstride + j];
  }

  void mul_matrix(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = src[i * colstride + j] * v;
  }

  void div_matrix_const(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = src[i * colstride + j] / v;
  }

  void div_const_matrix(double* dst, double* src, double v, size_t r, size_t c, size_t colstride){
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        dst[i * colstride + j] = v / src[i * colstride + j];
  }

  BinaryConstMtxOpTest(){}
  ~BinaryConstMtxOpTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));

    delete[] memblock;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock, devblockdst, memblock, memblockdst, memblockref;
  size_t rsz, csz;
};

TEST_F(BinaryConstMtxOpTest, Add1){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  add_matrix(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  add_const_2d_cuda_pd(devblockdst, devblock, 3., rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Add2){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  add_matrix(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  add_const_2d_cuda_pd(devblockdst, devblock, 3., rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Sub1){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  sub_matrix_const(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_mc_2d_cuda_pd(devblockdst, devblock, 3., rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Sub2){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  sub_matrix_const(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_mc_2d_cuda_pd(devblockdst, devblock, 3., rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Sub3){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  sub_const_matrix(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_cm_2d_cuda_pd(devblockdst, 3., devblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Sub4){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  sub_const_matrix(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sub_cm_2d_cuda_pd(devblockdst, 3., devblock, rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Mul1){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  mul_matrix(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  emul_const_2d_cuda_pd(devblockdst, devblock, 3., rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Mul2){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  mul_matrix(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  emul_const_2d_cuda_pd(devblockdst, devblock, 3., rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Div1){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  div_matrix_const(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_mc_2d_cuda_pd(devblockdst, devblock, 3., rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Div2){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  div_matrix_const(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_mc_2d_cuda_pd(devblockdst, devblock, 3., rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Div3){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  div_const_matrix(memblockref, memblock, 3., rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_cm_2d_cuda_pd(devblockdst, 3., devblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

TEST_F(BinaryConstMtxOpTest, Div4){
  rsz = 128, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz - 10, csz - 9, rsz, csz);

  div_const_matrix(memblockref, memblock, 3., rsz - 10, csz - 9, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ediv_cm_2d_cuda_pd(devblockdst, 3., devblock, rsz - 10, csz - 9, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz * csz));
}

struct ReductionMtxOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0;
  }
  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(-100., 100.);

    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void max_coefficient(double* dst, double* mtx, size_t r, size_t c, size_t rowsize, size_t colsize){
    size_t ir = 0;
    for (; ir < r; ++ir){
      dst[ir] = mtx[ir * colsize];
      for (size_t ic = 1; ic < c; ++ic)
        if (mtx[ir * colsize + ic] > dst[ir])
          dst[ir] = mtx[ir * colsize + ic];
    }
    for (; ir < rowsize; ++ir)
      dst[ir] = 0.;
  }

  void min_coefficient(double* dst, double* mtx, size_t r, size_t c, size_t rowsize, size_t colsize){
    size_t ir = 0;
    for (; ir < r; ++ir){
      dst[ir] = mtx[ir * colsize];
      for (size_t ic = 1; ic < c; ++ic)
        if (mtx[ir * colsize + ic] < dst[ir])
          dst[ir] = mtx[ir * colsize + ic];
    }
    for (; ir < rowsize; ++ir)
      dst[ir] = 0.;
  }

  void sum(double* dst, double* mtx, size_t r, size_t c, size_t rowsize, size_t colsize){
    size_t ir = 0;
    for (; ir < r; ++ir){
      dst[ir] = 0.;
      for (size_t ic = 0; ic < c; ++ic)
        dst[ir] += mtx[ir * colsize + ic];
    }
    for (; ir < rowsize; ++ir)
      dst[ir] = 0.;
  }

  void mean(double* dst, double* mtx, size_t r, size_t c, size_t rowsize, size_t colsize){
    sum(dst, mtx, r, c, rowsize, colsize);
    for (size_t i = 0; i < r; ++i)
      dst[i] /= (double)c;
  }

  void l2loss(double* dst, double* mtx, double reg, size_t r, size_t c){
    double square_sum = 0.;
    for (size_t i = 0; i < r * c; ++i)
      square_sum += mtx[i] * mtx[i];
    *dst = square_sum * 0.5 * reg;
  }

  void softmax(double* dst, const double* src, size_t r, size_t c, size_t colstride){
    for (size_t ir = 0; ir < r; ++ir){
      double sum = 0.;
      for (size_t ic = 0; ic < c; ++ic)
        sum += exp(src[ir * colstride + ic]);
      for (size_t ic = 0; ic < c; ++ic)
        dst[ir * colstride + ic] = exp(src[ir * colstride + ic]) / sum;
    }
  }

  ReductionMtxOpTest(){}
  ~ReductionMtxOpTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));
    CUDADBG(cudaFree(devblocksp));

    delete[] memblock;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock, devblockdst, devblocksp, memblock, memblockdst, memblockref;
  size_t rsz, csz;
};

TEST_F(ReductionMtxOpTest, MaxCoeff1){
  rsz = 4, csz = 1024;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff2){
  rsz = 4, csz = 512;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff3){
  rsz = 4, csz = 256;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff4){
  rsz = 4, csz = 64;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff5){
  rsz = 32, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff6){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff7){
  rsz = 16, csz = 256;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff8){
  rsz = 4, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used here

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff9){
  rsz = 4, csz = 4096;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 4)); //require 4 * 4 scratchpad

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff10){
  rsz = 32, csz = 8192;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 64)); //require 32 * 64 scratchpad

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff11){
  rsz = 32, csz = 2097152UL;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 16384 + sizeof(double) * rsz * 128)); //require 32 * 16384 + 32 * 128 scratchpad, two layers

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff12){
  rsz = 16, csz = 16777216UL;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 65536 + sizeof(double) * rsz * 256)); //require 16 * 65536 + 16 * 256 scratchpad, two layers

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff13){
  rsz = 64, csz = 16384;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 128)); //require 64 * 128 scratchpad, 1 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblock, rsz, csz, rsz, csz);

  max_coefficient(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff14){
  rsz = 32, csz = 32;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 2, csz - 20, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  max_coefficient(memblockref, memblock, rsz - 2, csz - 20, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz - 2, csz - 20, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, MaxCoeff15){
  rsz = 32, csz = 16384;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 128));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 2, csz - 20, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  max_coefficient(memblockref, memblock, rsz - 2, csz - 20, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_2d_cuda_pd(devblockdst, devblock, rsz - 2, csz - 20, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(ReductionMtxOpTest, Sum1){
  rsz = 4, csz = 2048;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 16));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Sum2){
  rsz = 32, csz = 256;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 2));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Sum3){
  rsz = 16, csz = 256;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 3, csz - 30, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz - 3, csz - 30, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz - 3, csz - 30, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Sum4){
  rsz = 32, csz = 65536;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 512)); //1 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz, csz, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Sum5){
  rsz = 32, csz = 65536;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 512)); //1 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 1, csz - 17, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz - 1, csz - 17, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz - 1, csz - 17, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Sum6){
  rsz = 32, csz = 16777216UL;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 131072 + sizeof(double) * rsz * 1024 + sizeof(double) * rsz * 8)); //3 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 3, csz - 3, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  sum(memblockref, memblock, rsz - 3, csz - 3, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_sum_2d_cuda_pd(devblockdst, devblock, rsz - 3, csz - 3, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, Mean1){
  rsz = 32, csz = 16777216UL;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * 4 * 32));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 131072 + sizeof(double) * rsz * 1024 + sizeof(double) * rsz * 8)); //3 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz];
  memblockref = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, rsz - 3, csz - 3, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., 4, 32, 4, 32);

  mean(memblockref, memblock, rsz - 3, csz - 3, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  row_mean_2d_cuda_pd(devblockdst, devblock, rsz - 3, csz - 3, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < rsz; ++i)
    EXPECT_NEAR(memblockref[i], memblockdst[i], 0.001);
}

TEST_F(ReductionMtxOpTest, L2loss1){
  rsz = 4, csz = 1024;
  double reg = 0.0001;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double))); //not used

  memblock = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblockref, 1);
  init_matrix(memblock, rsz, csz, rsz, csz);

  l2loss(memblockref, memblock, reg, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  loss_l2_1d_cuda_pd(devblockdst, devblock, reg, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(memblockref[0], memblockdst[0], 0.001);
}

TEST_F(ReductionMtxOpTest, L2loss2){
  rsz = 64, csz = 2048;
  double reg = 0.0001;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * csz / CUDA_MAX_TSZ / CUDA_SLICE_SZ)); //1 layer

  memblock = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblockref, 1);
  init_matrix(memblock, rsz, csz, rsz, csz);

  l2loss(memblockref, memblock, reg, rsz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  loss_l2_1d_cuda_pd(devblockdst, devblock, reg, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(memblockref[0], memblockdst[0], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax1){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz)); //rowsum size

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax2){
  rsz = 4, csz = 2048;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz)); //rowsum size

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax3){
  rsz = 32, csz = 512;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 32)); //rowsum size and 1 layer buffer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax4){
  rsz = 32, csz = 65536;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 256)); //rowsum size and 1 layer buffer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax5){
  rsz = 32, csz = 8388608UL;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 65536 + sizeof(double) * rsz * 512 + sizeof(double) * rsz * 4)); //rowsum size and 3 layer buffer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, rsz, csz, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

TEST_F(ReductionMtxOpTest, Softmax6){
  rsz = 32, csz = 8388608UL;
  size_t r = rsz - 2, c = csz - 19;
  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 65536 + sizeof(double) * rsz * 512 + sizeof(double) * rsz * 4)); //rowsum size and 3 layer buffer

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  softmax(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  softmax_row_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.001);
}

struct SingleMtxOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0;
  }
  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(-100., 100.);

    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void sigmoid(double* dst, double* src, size_t r, size_t c, size_t colsize){
    for (size_t ir = 0; ir < r; ++ir)
      for (size_t ic = 0; ic < c; ++ic)
        dst[ir * colsize + ic] = 1. / (1. + exp(src[ir * colsize + ic] * -1.));
  }

  void tanh(double* dst, double* src, size_t r, size_t c, size_t colsize){
    for (size_t ir = 0; ir < r; ++ir)
      for (size_t ic = 0; ic < c; ++ic)
        dst[ir * colsize + ic] = std::tanh(src[ir * colsize + ic]);
  }

  void relu(double* dst, double* src, size_t r, size_t c, size_t colsize){
    for (size_t ir = 0; ir < r; ++ir)
      for (size_t ic = 0; ic < c; ++ic)
        dst[ir * colsize + ic] = std::max(src[ir * colsize + ic], 0.);
  }

  SingleMtxOpTest(){}
  ~SingleMtxOpTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));

    delete[] memblock;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock, devblockdst, memblock, memblockdst, memblockref;
  size_t rsz, csz, r, c;
};

TEST_F(SingleMtxOpTest, Sigmoid1){
  rsz = 4, csz = 1024;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  sigmoid(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sigmoid_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Sigmoid2){
  rsz = 32, csz = 128;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  sigmoid(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sigmoid_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Sigmoid3){
  rsz = 32, csz = 128;
  r = rsz - 3, c = csz - 19;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  sigmoid(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sigmoid_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Sigmoid4){
  rsz = 128, csz = 1024;
  r = rsz - 1, c = csz - 11;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  sigmoid(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  sigmoid_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Tanh1){
  rsz = 4, csz = 1024;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  tanh(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  tanh_1d_cuda_pd(devblockdst, devblock, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Tanh2){
  rsz = 96, csz = 2048;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  tanh(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  tanh_1d_cuda_pd(devblockdst, devblock, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Relu1){
  rsz = 4, csz = 1024;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  relu(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  relu_1d_cuda_pd(devblockdst, devblock, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

TEST_F(SingleMtxOpTest, Relu2){
  rsz = 128, csz = 512;
  r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockdst = new double[rsz * csz];
  memblockref = new double[rsz * csz];

  init_matrix(memblockref, rsz * csz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz, csz, rsz, csz);

  relu(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  relu_1d_cuda_pd(devblockdst, devblock, rsz, csz);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz * csz, cudaMemcpyDeviceToHost));

  for (size_t ir = 0; ir < rsz; ++ir)
    for (size_t ic = 0; ic < csz; ++ic)
      EXPECT_NEAR(memblockref[ir * csz + ic], memblockdst[ir * csz + ic], 0.0001);
}

struct RowCoeffIdxOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0;
  }
  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(-100., 100.);

    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void max_row_coeff_idx(double* dst, double* src, size_t r, size_t c, size_t colstride){
    for (size_t ir = 0; ir < r; ++ir){
      size_t idx = 0;
      for (size_t ic = 1; ic < c; ++ic)
        if (src[ir * colstride + ic] > src[ir * colstride + idx])
          idx = ic;
      dst[ir] = idx;
    }
  }

  void min_row_coeff_idx(double* dst, double* src, size_t r, size_t c, size_t colstride){
    for (size_t ir = 0; ir < r; ++ir){
      size_t idx = 0;
      for (size_t ic = 1; ic < c; ++ic)
        if (src[ir * colstride + ic] < src[ir * colstride + idx])
          idx = ic;
      dst[ir] = idx;
    }
  }

  RowCoeffIdxOpTest(){}
  ~RowCoeffIdxOpTest(){
    CUDADBG(cudaFree(devblock));
    CUDADBG(cudaFree(devblockdst));
    CUDADBG(cudaFree(devblocksp));

    delete[] memblock;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock, devblockdst, devblocksp, memblock, memblockdst, memblockref;
  size_t rsz, csz;
};

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx1){
  rsz = 32, csz = 128;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx2){
  rsz = 32, csz = 256;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 2)); //1 layer

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx3){
  rsz = 32, csz = 16384;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 128)); //1 layer

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx4){
  rsz = 32, csz = 2097152UL;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 128 * 128 + sizeof(double) * rsz * 128)); //2 layer

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx5){
  rsz = 128, csz = 2097152UL;
  size_t r = rsz - 3, c = csz - 1;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * 128 * 128 + sizeof(double) * rsz * 128)); //2 layer

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz / 32, 32, rsz / 32, 32);

  max_row_coeff_idx(memblockref, memblock, r, c, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx6){
  rsz = 128, csz = 512;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz / 32, 32, rsz / 32, 32);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}

TEST_F(RowCoeffIdxOpTest, MaxRowCoeffIdx7){
  rsz = 512, csz = 512;
  size_t r = rsz, c = csz;

  CUDADBG(cudaMalloc(&devblock, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double) * rsz));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz * csz));

  memblock = new double[rsz * csz];
  memblockref = new double[rsz];
  memblockdst = new double[rsz];

  init_matrix(memblockref, rsz);
  init_matrix(memblock, r, c, rsz, csz);
  const_init_2d_cuda_pd(devblockdst, 0., rsz / 32, 32, rsz / 32, 32);

  max_row_coeff_idx(memblockref, memblock, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock, memblock, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  max_row_coeff_idx_2d_cuda_pd(devblockdst, devblock, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double) * rsz, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(0 == memcmp(memblockref, memblockdst, sizeof(double) * rsz));
}


struct LossOpTest : testing::Test {
  void init_matrix(double* m, size_t sz){
    for (size_t i = 0; i < sz; ++i)
      m[i] = 0;
  }

  void init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(-100., 100.);

    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void ec_init_matrix(double* m, size_t r, size_t c, size_t rowsize, size_t colsize){
    default_random_engine& eng = ML::get_default_random_engine();
    uniform_real_distribution<double> dist(0., 1.);

    init_matrix(m, rowsize * colsize);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        m[i * colsize + j] = dist(eng);
  }

  void mse_loss(double* dst, double* o, double* y, double rows, size_t sz){
    double ret = 0.;
    for (size_t i = 0; i < sz; ++i){
      double v = o[i] - y[i];
      ret += v * v;
    }
    *dst = ret / rows;
  }

  void mse_accuracy(double* dst, double* o, double* y, double rows, double sz){
    double ret = 0.;
    for (size_t i = 0; i < sz; ++i){
      double v = o[i] - y[i];
      ret += v * v;
    }
    *dst = sqrt(ret) / (rows * 2.);
  }

  void msvm_loss(double* dst, double* o, double* y, double f, size_t rows, size_t cols, size_t colstride){
    double loss = 0.;
    for (size_t ir = 0; ir < rows; ++ir){
      size_t midx = 0;
      for (size_t ic = 1; ic < cols; ++ic){
        if (y[ir * colstride + ic] > y[ir * colstride + midx])
          midx = ic;
      }
      double v = o[ir * colstride + midx];
      for (size_t ic = 0; ic < cols; ++ic){
        double e = o[ir * colstride + ic] - v + f;
        if (e < 0.) e = 0.;
        loss += e * e;
      }
      loss -= f * f;
    }
    *dst = loss / (double)rows;
  }

  void accuracy(double* dst, double* o, double* y, size_t rows, size_t cols, size_t colstride){
    size_t count = 0;
    for (size_t ir = 0; ir < rows; ++ir){
      size_t oidx = 0, yidx = 0;
      for (size_t ic = 1; ic < cols; ++ic){
        if (o[ir * colstride + ic] > o[ir * colstride + oidx])
          oidx = ic;
        if (y[ir * colstride + ic] > y[ir * colstride + yidx])
          yidx = ic;
      }
      if (oidx == yidx) count++;
    }
    *dst = (double)count / (double)rows;
  }

  void cross_entropy_loss(double* dst, double* o, double* y, size_t rows, size_t cols, size_t colstride){
    double loss = 0.;
    for (size_t ir = 0; ir < rows; ++ir){
      size_t yidx = 0;
      for (size_t ic = 1; ic < cols; ++ic)
        if (y[ir * colstride + ic] > y[ir * colstride + yidx])
          yidx = ic;
      loss += -1. * log(o[ir * colstride + yidx]);
    }
    *dst = loss / (double)rows;
  }

  LossOpTest(){}
  ~LossOpTest(){
    CUDADBG(cudaFree(devblock1));
    CUDADBG(cudaFree(devblock2));
    CUDADBG(cudaFree(devblockdst));
    CUDADBG(cudaFree(devblocksp));

    delete[] memblock1;
    delete[] memblock2;
    delete[] memblockdst;
    delete[] memblockref;
  }

  doubleptr devblock1, devblock2, devblockdst, devblocksp, memblock1, memblock2, memblockdst, memblockref;
  size_t rsz, csz;
};

TEST_F(LossOpTest, MSELoss1){
  rsz = 32, csz = 32;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSELoss2){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSELoss3){
  rsz = 4, csz = 1024;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSELoss4){
  rsz = 32, csz = 256;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * 2));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSELoss5){
  rsz = 32, csz = 1024;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * 8));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSELoss6){
  rsz = 32, csz = 524288;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * 4096));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_loss(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_loss_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSEAccuracy1){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double)));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_accuracy(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_accuracy_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSEAccuracy2){
  rsz = 32, csz = 524288;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * 4096));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  mse_accuracy(memblockref, memblock1, memblock2, rsz, rsz * csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  mse_accuracy_1d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss1){
  rsz = 32, csz = 128;
  double f = 0.01;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss2){
  rsz = 32, csz = 512;
  double f = 0.001;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 4));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss3){
  rsz = 128, csz = 512;
  double f = 0.01;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 4));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss4){
  rsz = 128, csz = 16384;
  double f = 0.01;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 128 + sizeof(double) * 16));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss5){
  rsz = 4096, csz = 16384;
  double f = 0.01;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 128 + sizeof(double) * 16));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMLoss6){
  rsz = 4096, csz = 16384;
  size_t r = rsz - 2, c = csz - 3;
  double f = 0.01;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 128 + sizeof(double) * 16));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, r, c, rsz, csz);
  init_matrix(memblock2, r, c, rsz, csz);

  msvm_loss(memblockref, memblock1, memblock2, f, r, c, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  msvm_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, f, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMAccuracy1){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  accuracy(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  accuracy_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMAccuracy2){
  rsz = 32, csz = 512;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  accuracy(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  accuracy_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMAccuracy3){
  rsz = 4, csz = 1024;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  accuracy(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  accuracy_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMAccuracy4){
  rsz = 128, csz = 1024;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz * 7));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, rsz, csz, rsz, csz);
  init_matrix(memblock2, rsz, csz, rsz, csz);

  accuracy(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  accuracy_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, MSVMAccuracy5){
  rsz = 4096, csz = 16384;
  size_t r = rsz - 1, c = csz - 29;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz + sizeof(double) * rsz * 127 + sizeof(double) * rsz * 16));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  init_matrix(memblock1, r, c, rsz, csz);
  init_matrix(memblock2, r, c, rsz, csz);

  accuracy(memblockref, memblock1, memblock2, r, c, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  accuracy_2d_cuda_pd(devblockdst, devblock1, devblock2, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss1){
  rsz = 32, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, rsz, csz, rsz, csz);
  ec_init_matrix(memblock2, rsz, csz, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss2){
  rsz = 2048, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, rsz, csz, rsz, csz);
  ec_init_matrix(memblock2, rsz, csz, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss3){
  rsz = 32, csz = 128;
  size_t r = rsz - 2, c = csz - 22;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, r, c, rsz, csz);
  ec_init_matrix(memblock2, r, c, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, r, c, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss4){
  rsz = 1048576UL, csz = 128;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, rsz, csz, rsz, csz);
  ec_init_matrix(memblock2, rsz, csz, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss5){
  rsz = 4096, csz = 1024;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 8 + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, rsz, csz, rsz, csz);
  ec_init_matrix(memblock2, rsz, csz, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, rsz, csz, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, rsz, csz, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}

TEST_F(LossOpTest, CrossEntropyLoss6){
  rsz = 4096, csz = 1024;
  size_t r = rsz - 1, c = csz - 1;
  CUDADBG(cudaMalloc(&devblock1, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblock2, sizeof(double) * rsz * csz));
  CUDADBG(cudaMalloc(&devblockdst, sizeof(double)));
  CUDADBG(cudaMalloc(&devblocksp, sizeof(double) * rsz + sizeof(double) * rsz * 8 + sizeof(double) * rsz));

  memblock1 = new double[rsz * csz];
  memblock2 = new double[rsz * csz];
  memblockdst = new double[1];
  memblockref = new double[1];

  ec_init_matrix(memblock1, r, c, rsz, csz);
  ec_init_matrix(memblock2, r, c, rsz, csz);

  cross_entropy_loss(memblockref, memblock1, memblock2, r, c, csz);

  CUDADBG(cudaMemcpy(devblock1, memblock1, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));
  CUDADBG(cudaMemcpy(devblock2, memblock2, sizeof(double) * rsz * csz, cudaMemcpyHostToDevice));

  ce_loss_2d_cuda_pd(devblockdst, devblock1, devblock2, r, c, rsz, csz, devblocksp);

  CUDADBG(cudaMemcpy(memblockdst, devblockdst, sizeof(double), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(*memblockref, *memblockdst, 0.001);
}


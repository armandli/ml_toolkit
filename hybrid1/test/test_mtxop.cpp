#include <gtest/gtest.h>

#include <ml_common.h>
#include <ml_mtxop.h>

using namespace std;
using namespace ML::MTXOP;

struct TestInit : testing::Test {
  TestInit(){
    rows = rowstride = MTX_BLOCK_RSZ * 8;
    cols = colstride = MTX_BLOCK_CSZ * 10;
    buffer1 = new double[rowstride * colstride];
  }
  ~TestInit(){
    delete[] buffer1;
  }

  size_t rows, cols, rowstride, colstride;
  double* buffer1;
};

TEST_F(TestInit, UniformInitBlock1){
  rnd_uniform_init_2d_mtxop_pd(buffer1, -10., 10., rows, cols, rowstride, colstride);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= -10.);
    EXPECT_TRUE(buffer1[i] <=  10.);
  }
}

TEST_F(TestInit, UniformInitBlock2){
  rnd_uniform_init_2d_mtxop_pd(buffer1, 5., 15., rows, cols, rowstride, colstride);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 5.);
    EXPECT_TRUE(buffer1[i] <= 15.);
  }
}


TEST_F(TestInit, GaussianInitBlock1){
  rnd_normal_init_2d_mtxop_pd(buffer1, 10., 20., MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10, MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 10. - 20. * 5);
    EXPECT_TRUE(buffer1[i] <= 10. + 20. * 5);
  }
}

TEST_F(TestInit, GaussianInitBlock2){
  rnd_normal_init_2d_mtxop_pd(buffer1, 10., 1., MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10, MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 10. - 1. * 5);
    EXPECT_TRUE(buffer1[i] <= 10. + 1. * 5);
  }
}



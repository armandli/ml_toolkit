#include <gtest/gtest.h>

#include <cstdlib>

#include <ml_common.h>
#include <ml_mtxop.h>

using namespace std;
using namespace ML::MTXOP;

struct TestInit : testing::Test {
  TestInit(){
    rows = rowstride = MTX_BLOCK_RSZ * 8;
    cols = colstride = MTX_BLOCK_CSZ * 10;
    buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  }
  ~TestInit(){
    free(buffer1);
  }

  size_t rows, cols, rowstride, colstride;
  double* buffer1;
};

TEST_F(TestInit, UniformInitBlock1){
  rnd_uniform_init_2d_mtxop_pd(buffer1, -10., 10., rows, cols, colstride);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= -10.);
    EXPECT_TRUE(buffer1[i] <=  10.);
  }
}

TEST_F(TestInit, UniformInitBlock2){
  rnd_uniform_init_2d_mtxop_pd(buffer1, 5., 15., rows, cols, colstride);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 5.);
    EXPECT_TRUE(buffer1[i] <= 15.);
  }
}


TEST_F(TestInit, GaussianInitBlock1){
  rnd_normal_init_2d_mtxop_pd(buffer1, 10., 20., MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10, MTX_BLOCK_CSZ * 10);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 10. - 20. * 5);
    EXPECT_TRUE(buffer1[i] <= 10. + 20. * 5);
  }
}

TEST_F(TestInit, GaussianInitBlock2){
  rnd_normal_init_2d_mtxop_pd(buffer1, 10., 1., MTX_BLOCK_RSZ * 8, MTX_BLOCK_CSZ * 10, MTX_BLOCK_CSZ * 10);

  for (size_t i = 0; i < rowstride * colstride; ++i){
    EXPECT_TRUE(buffer1[i] >= 10. - 1. * 5);
    EXPECT_TRUE(buffer1[i] <= 10. + 1. * 5);
  }
}

struct CostTest : testing::Test {
  CostTest(){}
  ~CostTest(){
    free(buffer1);
    free(buffer2);
    delete dst1;
  }

  void ce_init(double* b, size_t rows, size_t cols, size_t colstride, size_t idx_offset){
    double val1 = 0.1 / (double)cols;
    double val2 = 0.9;

    for (size_t ir = 0, idx = idx_offset; ir < rows; ++ir, ++idx)
      for (size_t ic = 0; ic < cols; ++ic){
        if (idx >= cols) idx = 0;
        if (ic == idx)
          b[ir * colstride + ic] = val1 + val2;
        else
          b[ir * colstride + ic] = val1;
      }
  }

  size_t rows, cols, rowstride, colstride;
  double* buffer1, * buffer2;
  double* dst1;
};

TEST_F(CostTest, CELoss1){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 0);
  ce_init(buffer2, rows, cols, colstride, 1);

  ce_loss_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, -1. * std::log(0.1 / (double)cols));
}

TEST_F(CostTest, CELoss2){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 3);
  ce_init(buffer2, rows, cols, colstride, 3);

  ce_loss_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, -1. * std::log(0.1 / (double)cols + 0.9));
}

TEST_F(CostTest, CELoss3){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 3);
  ce_init(buffer2, rows, cols, colstride, 5);

  ce_loss_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, -1. * std::log(0.1 / (double)cols));
}


TEST_F(CostTest, CEAccuracy1){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 0);
  ce_init(buffer2, rows, cols, colstride, 1);

  ce_accuracy_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, 0.);
}

TEST_F(CostTest, CEAccuracy2){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 3);
  ce_init(buffer2, rows, cols, colstride, 3);

  ce_accuracy_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, 1.);
}

TEST_F(CostTest, CEAccuracy3){
  rows = rowstride = MTX_BLOCK_RSZ * 8;
  cols = colstride = MTX_BLOCK_CSZ * 10;
  buffer1 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  buffer2 = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * rowstride * colstride);
  dst1 = new double(0.);

  ce_init(buffer1, rows, cols, colstride, 2);
  ce_init(buffer2, rows, cols, colstride, 6);

  ce_accuracy_2d_mtxop_pd(dst1, buffer1, buffer2, rows, cols, colstride);
  EXPECT_DOUBLE_EQ(*dst1, 0.);
}


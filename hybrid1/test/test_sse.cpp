#include <gtest/gtest.h>

#include <ctime>
#include <random>
#include <limits>

#include <ml_common.h>
#include <ml_sse.h>

using namespace std;
using namespace ML::SSE;

double random_double(){
  static default_random_engine eng(time(0));
  static uniform_real_distribution<double> dist(-100., +100.);
  return dist(eng);
}

double random_double_softmax(){
  static default_random_engine eng(time(0));
  static uniform_real_distribution<double> dist(0.,1.);
  return dist(eng);
}

TEST(SSE, ConstInitBlock){
  double* buffer1 = new double[MTX_BLOCK_RSZ * 10 * 10];
  double* buffer2 = new double[MTX_BLOCK_RSZ * 10 * 10];
  double* expected1 = new double[MTX_BLOCK_RSZ * 10 * 10];
  double* expected2 = new double[MTX_BLOCK_RSZ * 10 * 10];
  size_t blksz = 20;

  const_init_2d_sse_pd(buffer1, 10., 17, 18, blksz, blksz);
  const_init_2d_sse_pd(buffer2, 33., 20, 20, blksz, blksz);

  for (size_t i = 0; i < 20; ++i)
    for (size_t j = 0; j < 20; ++j){
      expected1[i * 20 + j] = 0.;
      expected2[i * 20 + j] = 33.;
    }

  for (size_t i = 0; i < 17; ++i)
    for (size_t j = 0; j < 18; ++j)
      expected1[i * 20 + j] = 10.;

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * MTX_BLOCK_RSZ * 10 * 10));
  EXPECT_TRUE(0 == memcmp(buffer2, expected2, sizeof(double) * MTX_BLOCK_RSZ * 10 * 10));

  delete[] buffer1;
  delete[] buffer2;
  delete[] expected1;
  delete[] expected2;
}

TEST(SSE, TransposeBlock1){
  double* buffer1 = new double[MTX_BLOCK_RSZ * 8 * 8];
  double* buffer2 = new double[MTX_BLOCK_RSZ * 3 * 4];
  double* buffer3 = new double[MTX_BLOCK_RSZ * 4 * 3];
  double* dst1 = new double[MTX_BLOCK_RSZ * 8 * 8];
  double* dst2 = new double[MTX_BLOCK_RSZ * 3 * 4];
  double* dst3 = new double[MTX_BLOCK_RSZ * 4 * 3];
  double* expected1 = new double[MTX_BLOCK_RSZ * 8 * 8];
  double* expected2 = new double[MTX_BLOCK_RSZ * 3 * 4];
  double* expected3 = new double[MTX_BLOCK_RSZ * 4 * 3];

  for (size_t i = 0; i < MTX_BLOCK_RSZ * 8 * 8; ++i){
    buffer1[i] = (double)i;
    expected1[i] = (double)i;
  }
  for (size_t i = 0; i < MTX_BLOCK_RSZ * 3 * 4; ++i){
    buffer2[i] = (double)i;
    expected2[i] = (double)i;
  }
  for (size_t i = 0; i < MTX_BLOCK_RSZ * 4 * 3; ++i){
    buffer3[i] = (double)i;
    expected3[i] = (double)i;
  }

  transpose4x4_2d_sse_pd(dst1, buffer1, 16, 16, 16, 16);
  transpose4x4_2d_sse_pd(dst2, buffer2, 12, 4, 12, 4);
  transpose4x4_2d_sse_pd(dst3, buffer3, 4, 12, 4, 12);

  for (size_t i = 0; i < 16; ++i)
    for (size_t j = 0; j < 16; ++j)
      expected1[j * 16 + i] = i * 16 + j;

  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 12; ++j)
      expected2[j * 4 + i] = i * 12 + j;

  for (size_t i = 0; i < 12; ++i)
    for (size_t j = 0; j < 4; ++j)
      expected3[j * 12 + i] = i * 4 + j;

  EXPECT_TRUE(0 == memcmp(dst1, expected1, sizeof(double) * 8 * 8 * MTX_BLOCK_RSZ));
  EXPECT_TRUE(0 == memcmp(dst2, expected2, sizeof(double) * 3 * 4 * MTX_BLOCK_RSZ));
  EXPECT_TRUE(0 == memcmp(dst3, expected3, sizeof(double) * 4 * 3 * MTX_BLOCK_RSZ));

  delete[] buffer1;
  delete[] dst1;
  delete[] expected1;
  delete[] buffer2;
  delete[] dst2;
  delete[] expected2;
  delete[] buffer3;
  delete[] dst3;
  delete[] expected3;
}

TEST(SSE, TransposeBlock2){
  double* buffer1 = new double[52 * 32];
  double* dst1 = new double[32 * 64];
  double* expected1 = new double[32 * 64];
  for (size_t i = 0; i < 52 * 32; ++i)
    buffer1[i] = i;
  for (size_t ir = 0; ir < 32; ++ir)
    for (size_t ic = 0; ic < 52; ++ic)
      expected1[ir * 64 + ic] = buffer1[ic * 32 + ir];

  transpose4x4_2d_sse_pd(dst1, buffer1, 32, 52, 32, 64);

  for (size_t ir = 0; ir < 32; ++ir)
    for (size_t ic = 0; ic < 52; ++ic)
      EXPECT_DOUBLE_EQ(dst1[ir * 64 + ic], expected1[ir * 64 + ic]);
}

TEST(SSE, TransposeBlock3){
  double* buffer1 = new double[52 * 32]; //50 * 4
  double* dst1 = new double[4 * 64];     //4  * 50
  double* expected1 = new double[32 * 64];
  for (size_t ir = 0; ir < 50; ++ir)
    for (size_t ic = 0; ic < 4; ++ic)
      buffer1[ir * 32 + ic] = ir * 4 + ic;
  for (size_t ir = 0; ir < 50; ++ir)
    for (size_t ic = 0; ic < 4; ++ic)
      expected1[ic * 64 + ir] = buffer1[ir * 32 + ic];

  transpose4x4_2d_sse_pd(dst1, buffer1, 4, 52, 32, 64);

  for (size_t ir = 0; ir < 4; ++ir)
    for (size_t ic = 0; ic < 50; ++ic)
      EXPECT_DOUBLE_EQ(expected1[ir * 64 + ic], dst1[ir * 64 + ic]);
}

struct SSESimpleOperation : testing::Test {
  SSESimpleOperation(){
    buffer1 = new double[MTX_BLOCK_RSZ * 12 * 12];
    buffer2 = new double[MTX_BLOCK_RSZ * 12 * 12];
    expected1 = new double[MTX_BLOCK_RSZ * 12 * 12];
    buffer3 = new double[MTX_BLOCK_RSZ * 8];
    buffer4 = new double[MTX_BLOCK_RSZ * 8];
    expected2 = new double[MTX_BLOCK_RSZ * 8];

    for (size_t i = 0; i < 24; ++i)
      for (size_t j = 0; j < 24; ++j){
        buffer1[i * 24 + j] = i * 24 + j;
        buffer2[i * 24 + j] = i * 24 + j + 1;
      }

    for (size_t i = 0; i < 8; ++i)
      for (size_t j = 0; j < 4; ++j){
        buffer3[i * 4 + j] = i * 4 + j;
        buffer4[i * 4 + j] = j * 4 + i;
      }
  }

  ~SSESimpleOperation(){
    delete[] buffer1;
    delete[] buffer2;
    delete[] buffer3;
    delete[] buffer4;
    delete[] expected1;
    delete[] expected2;
  }

  double* buffer1;
  double* buffer2;
  double* buffer3;
  double* buffer4;
  double* expected1;
  double* expected2;
};

TEST_F(SSESimpleOperation, AddBlock){
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = i * 24 + j + i * 24 + j + 1;

  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 4; ++j)
      expected2[i * 4 + j] = i * 4 + j + j * 4 + i;

  add_1d_sse_pd(buffer1, buffer1, buffer2, 24, 24);
  add_1d_sse_pd(buffer3, buffer3, buffer4, 8, 4);

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
  EXPECT_TRUE(0 == memcmp(buffer3, expected2, sizeof(double) * 8 * 4));
}

TEST_F(SSESimpleOperation, AddConst){
  double b = 13.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = i * 24 + j + b;

  add_const_2d_sse_pd(buffer1, buffer1, b, 24, 24, 24);

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

TEST_F(SSESimpleOperation, SubBlock){
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = ((double)i * 24 + j) - ((double)i * 24 + j + 1);

  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 4; ++j)
      expected2[i * 4 + j] = ((double)i * 4 + j) - ((double)j * 4 + i);

  sub_1d_sse_pd(buffer1, buffer1, buffer2, 24, 24);
  sub_1d_sse_pd(buffer3, buffer3, buffer4, 8, 4);

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
  EXPECT_TRUE(0 == memcmp(buffer3, expected2, sizeof(double) * 8 * 4));
}

TEST_F(SSESimpleOperation, SubConst1){
  double b = 13.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = ((double)i * 24 + j) - b;

  sub_mc_2d_sse_pd(buffer1, buffer1, b, 24, 24, 24);
  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

TEST_F(SSESimpleOperation, SubConst2){
  double b = 13.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = b - ((double)i * 24 + j);

  sub_cm_2d_sse_pd(b, buffer1, buffer1, 24, 24, 24);
  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

TEST_F(SSESimpleOperation, ElemMulBlock){
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = (i * 24 + j) * (i * 24 + j + 1);

  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 4; ++j)
      expected2[i * 4 + j] = (i * 4 + j) * (j * 4 + i);

  emul_1d_sse_pd(buffer1, buffer1, buffer2, 24, 24);
  emul_1d_sse_pd(buffer3, buffer3, buffer4, 8, 4);

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
  EXPECT_TRUE(0 == memcmp(buffer3, expected2, sizeof(double) * 8 * 4));
}

TEST_F(SSESimpleOperation, ElemMulConst){
  double b = 33.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = (i * 24 + j) * b;

  emul_const_2d_sse_pd(buffer1, buffer1, b, 24, 24);
  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

TEST_F(SSESimpleOperation, ElemDivBlock){
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = ((double)i * 24 + j) / ((double)i * 24 + j + 1);

  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 4; ++j)
      expected2[i * 4 + j] = ((double)i * 4 + j) / ((double)j * 4 + i);

  ediv_2d_sse_pd(buffer1, buffer1, buffer2, 24, 24, 24);
  ediv_2d_sse_pd(buffer3, buffer3, buffer4, 8, 4, 4);

  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
  EXPECT_TRUE(0 == memcmp(buffer3, expected2, sizeof(double) * 8 * 4));
}

TEST_F(SSESimpleOperation, ElemDivConst1){
  double b = 7.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = ((double)i * 24 + j) / b;


  ediv_mc_2d_sse_pd(buffer1, buffer1, b, 24, 24, 24);
  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

TEST_F(SSESimpleOperation, ElemDivConst2){
  double b = 7.;
  for (size_t i = 0; i < 24; ++i)
    for (size_t j = 0; j < 24; ++j)
      expected1[i * 24 + j] = b / ((double)i * 24 + j);

  ediv_cm_2d_sse_pd(b, buffer1, buffer1, 24, 24, 24);
  EXPECT_TRUE(0 == memcmp(buffer1, expected1, sizeof(double) * 24 * 24));
}

void assign_zero(double* a, size_t sz){
  for (size_t i = 0; i < sz; ++i)
    a[i] = 0.;
}

struct SSEComplexReduction : testing::Test {
  SSEComplexReduction(){
    buffer1 = new double[MTX_BLOCK_RSZ * 12 * 12];
    buffer2 = new double[MTX_BLOCK_RSZ * 12 * 12];
    buffer5 = new double[MTX_BLOCK_RSZ * 12 * 12];
    expected1 = new double[MTX_BLOCK_RSZ * 12 * 12];
    buffer3 = new double[MTX_BLOCK_RSZ * 12 * 8];
    buffer4 = new double[MTX_BLOCK_RSZ * 12 * 8];
    expected2 = new double[MTX_BLOCK_RSZ * 12 * 8];

    assign_zero(buffer1, 24 * 24);
    assign_zero(buffer2, 24 * 24);
    assign_zero(buffer5, 24 * 24);
    assign_zero(expected1, 24 * 24);
    assign_zero(buffer3, 24 * 16);
    assign_zero(buffer4, 24 * 16);
    assign_zero(expected2, 24 * 16);
  }
  ~SSEComplexReduction(){
    delete[] buffer1;
    delete[] buffer2;
    delete[] expected1;
    delete[] buffer3;
    delete[] buffer4;
    delete[] expected2;
    delete[] buffer5;
  }
  double* buffer1;
  double* buffer2;
  double* buffer3;
  double* buffer4;
  double* buffer5;
  double* expected1;
  double* expected2;
};

TEST_F(SSEComplexReduction, MaxRowCoeffs){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 16; ++ir)
    for (size_t ic = 0; ic < 23; ++ic)
      buffer3[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir){
    double max = numeric_limits<double>::min();
    for (size_t ic = 0; ic < 21; ++ic)
      if (max < buffer1[ir * 24 + ic])
        max = buffer1[ir * 24 + ic];
    expected1[ir] = max;
  }

  for (size_t ir = 0; ir < 16; ++ir){
    double max = numeric_limits<double>::min();
    for (size_t ic = 0; ic < 23; ++ic)
      if (max < buffer3[ir * 24 + ic])
        max = buffer3[ir * 24 + ic];
    expected2[ir] = max;
  }

  max_row_coeffs_2d_sse_pd(buffer2, buffer1, 24, 21, 24);
  max_row_coeffs_2d_sse_pd(buffer4, buffer3, 16, 23, 24);

  EXPECT_TRUE(0 == memcmp(buffer2, expected1, sizeof(double) * 24));
  EXPECT_TRUE(0 == memcmp(buffer4, expected2, sizeof(double) * 16));
}

TEST_F(SSEComplexReduction, MinRowCoeffs){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 16; ++ir)
    for (size_t ic = 0; ic < 23; ++ic)
      buffer3[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir){
    double min = numeric_limits<double>::max();
    for (size_t ic = 0; ic < 21; ++ic)
      if (min > buffer1[ir * 24 + ic])
        min = buffer1[ir * 24 + ic];
    expected1[ir] = min;
  }

  for (size_t ir = 0; ir < 16; ++ir){
    double min = numeric_limits<double>::max();
    for (size_t ic = 0; ic < 23; ++ic)
      if (min > buffer3[ir * 24 + ic])
        min = buffer3[ir * 24 + ic];
    expected2[ir] = min;
  }

  min_row_coeffs_2d_sse_pd(buffer2, buffer1, 24, 21, 24);
  min_row_coeffs_2d_sse_pd(buffer4, buffer3, 16, 23, 24);

  EXPECT_TRUE(0 == memcmp(buffer2, expected1, sizeof(double) * 24));
  EXPECT_TRUE(0 == memcmp(buffer4, expected2, sizeof(double) * 16));
}

TEST_F(SSEComplexReduction, SumRows){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 16; ++ir)
    for (size_t ic = 0; ic < 23; ++ic)
      buffer3[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir){
    double s = 0.;
    for (size_t ic = 0; ic < 21; ++ic)
      s += buffer1[ir * 24 + ic];
    expected1[ir] = s;
  }

  for (size_t ir = 0; ir < 16; ++ir){
    double s = 0.;
    for (size_t ic = 0; ic < 23; ++ic)
      s += buffer3[ir * 24 + ic];
    expected2[ir] = s;
  }

  sum_rows_2d_sse_pd(buffer2, buffer1, 24, 24);
  sum_rows_2d_sse_pd(buffer4, buffer3, 16, 24);

  for (size_t i = 0; i < 24; ++i)
    EXPECT_NEAR(buffer2[i], expected1[i], 0.00000001);

  for (size_t i = 0; i < 16; ++i)
    EXPECT_NEAR(buffer4[i], expected2[i], 0.00000001);
}

TEST_F(SSEComplexReduction, SumAll){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  double expected = 0.;
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected += buffer1[ir * 24 + ic];

  double t;
  sum_all_1d_sse_pd(&t, buffer1, 24, 24);

  EXPECT_NEAR(expected, t, 0.00000001);
}

TEST_F(SSEComplexReduction, Sigmoid){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected1[ir * 24 + ic] = 1. / (1. + exp(buffer1[ir * 24 + ic] * -1.));

  sigmoid_2d_sse_pd(buffer2, buffer1, 24, 21, 24);

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.001);
}

TEST_F(SSEComplexReduction, Tanh){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected1[ir * 24 + ic] = tanh(buffer1[ir * 24 + ic]);

  tanh_1d_sse_pd(buffer2, buffer1, 24, 24);

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.001);
}

TEST_F(SSEComplexReduction, Relu){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected1[ir * 24 + ic] = buffer1[ir * 24 + ic] > 0. ? buffer1[ir * 24 + ic] : 0.;

  relu_1d_sse_pd(buffer2, buffer1, 24, 24);

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.001);
}

TEST_F(SSEComplexReduction, Drelu){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      buffer1[ir * 24 + ic] = random_double();
      buffer5[ir * 24 + ic] = random_double();
    }

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected1[ir * 24 + ic] = buffer5[ir * 24 + ic] > 0. ? buffer1[ir * 24 + ic] : 0.;

  drelu_1d_sse_pd(buffer2, buffer1, buffer5, 24, 24);

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.001);
}

TEST_F(SSEComplexReduction, L2Loss){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double();

  double reg = 0.5;
  double expected = 0.;
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      expected += buffer1[ir * 24 + ic] * buffer1[ir * 24 + ic];
  expected = expected * 0.5 * reg;

  double res;
  loss_l2_1d_sse_pd(&res, buffer1, reg, 24, 24);

  EXPECT_NEAR(res, expected, 0.00000001);
}

TEST_F(SSEComplexReduction, Softmax1){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 24; ++ic)
      buffer1[ir * 24 + ic] = random_double_softmax();

  for (size_t ir = 0; ir < 24; ++ir){
    double s = 0.;
    for (size_t ic = 0; ic < 24; ++ic)
      s += std::exp(buffer1[ir * 24 + ic]);
    for (size_t ic = 0; ic < 24; ++ic)
      expected1[ir * 24 + ic] = exp(buffer1[ir * 24 + ic]) / s;
  }

  softmax_r_2d_sse_pd(buffer2, buffer1, 24, 24, 24);
  
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 24; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.008); //SSE fexp only can handle sig fig to 0.01
}

TEST_F(SSEComplexReduction, Softmax2){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      buffer1[ir * 24 + ic] = random_double_softmax();

  for (size_t ir = 0; ir < 24; ++ir){
    double s = 0.;
    for (size_t ic = 0; ic < 21; ++ic)
      s += exp(buffer1[ir * 24 + ic]);
    for (size_t ic = 0; ic < 21; ++ic)
      expected1[ir * 24 + ic] = exp(buffer1[ir * 24 + ic]) / s;
  }

  softmax_r_2d_sse_pd(buffer2, buffer1, 24, 21, 24);
  
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.008); //SSE fexp only can handle sig fig to 0.01
}

TEST_F(SSEComplexReduction, MSELoss){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      buffer1[ir * 24 + ic] = random_double();
      buffer5[ir * 24 + ic] = random_double();
    }

  double s = 0.;
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      double v = buffer1[ir * 24 + ic] - buffer5[ir * 24 + ic];
      s += v * v;
    }
  s /= 24.;

  double r;
  mse_loss_1d_sse_pd(&r, buffer1, buffer5, 24, 24);

  EXPECT_NEAR(r, s, 0.00000001);
}

TEST_F(SSEComplexReduction, Deriviative){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      buffer1[ir * 24 + ic] = random_double();
      buffer5[ir * 24 + ic] = random_double();
    }

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      double v = (buffer1[ir * 24 + ic] - buffer5[ir * 24 + ic]) / 24.;
      if (isnan(v)) v = 0.;
      expected1[ir * 24 + ic] = v;
    }

  deriviative_row_1d_sse_pd(buffer2, buffer1, buffer5, 24, 24);

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 24; ++ic)
      EXPECT_NEAR(expected1[ir * 24 + ic], buffer2[ir * 24 + ic], 0.001);
}

TEST_F(SSEComplexReduction, CompareEqual){
  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      double v = random_double();
      buffer1[ir * 24 + ic] = v;
      buffer5[ir * 24 + ic] = v;
      buffer2[ir * 24 + ic] = random_double();
    }

  EXPECT_TRUE(block_cmp_equal_1d_sse_pd(buffer1, buffer5, 0.0000001, 24, 24));
  EXPECT_FALSE(block_cmp_equal_1d_sse_pd(buffer1, buffer2, 0.0000001, 24, 24));

  for (size_t ir = 0; ir < 24; ++ir)
    for (size_t ic = 0; ic < 21; ++ic){
      double v = random_double();
      buffer1[ir * 24 + ic] = v;
      buffer5[ir * 24 + ic] = v - 0.00000001;
    }
  EXPECT_TRUE(block_cmp_equal_1d_sse_pd(buffer1, buffer5, 0.0000001, 24, 24));
}

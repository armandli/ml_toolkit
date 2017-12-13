#include <gtest/gtest.h>

#include <cmath>

#include <ml_common.h>
#include <ml_matrix.h>
#include <ml_ssa.h>
#include <ml_mem_codegen.h>

using ML::Mtx;
using ML::MemArena;

void init(Mtx& m){
  for (size_t ir = 0; ir < m.rows(); ++ir)
    for (size_t ic = 0; ic < m.cols(); ++ic)
      m(ir, ic) = ir * m.cols() + ic;
}

void init2(Mtx& m){
  for (size_t ir = 0; ir < m.rows(); ++ir)
    for (size_t ic = 0; ic < m.cols(); ++ic)
      m(ir, ic) = ir * m.cols() + ic + 1;
}

void init_nan(Mtx& m){
  for (size_t ir = 0; ir < m.rows(); ++ir)
    for (size_t ic = 0; ic < m.cols(); ++ic)
      m(ir, ic) = nan("");
}

void init_random(Mtx& m){
  std::default_random_engine& eng = ML::get_default_random_engine();
  std::normal_distribution<double> dist(0., 0.5);
  for (size_t ir = 0; ir < m.rows(); ++ir)
    for (size_t ic = 0; ic < m.cols(); ++ic)
      m(ir, ic) = dist(eng);
}

struct ElemOp : testing::Test {
  ElemOp(){}
  ~ElemOp(){}

  MemArena arena;
};

TEST_F(ElemOp, Add1){
  Mtx a(16, 16), b(16, 16);
  init(a);
  init(b);

  Mtx c = a + b;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), ((double)ir * c.cols() + (double)ic) * 2.);
}

TEST_F(ElemOp, Add2){
  Mtx a(5000, 41), b(5000, 41);
  init(a);
  init(b);

  Mtx c = a + b;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), ((double)ir * c.cols() + (double)ic) * 2.);
}

TEST_F(ElemOp, Sub1){
  Mtx a(16, 16), b(16, 16);
  init(a);
  init(b);

  Mtx c = a - b;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), 0.);
}

TEST_F(ElemOp, Mul1){
  Mtx a(16, 16), b(16, 16);
  init(a);
  init(b);

  Mtx c = a * b;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), ((double)ir * c.cols() + (double)ic) * ((double)ir * c.cols() + (double)ic));
}

TEST_F(ElemOp, Div1){
  Mtx a(16, 16), b(16, 16);
  init2(a);
  init2(b);

  Mtx c = a / b;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), 1.);
}

TEST_F(ElemOp, MulMC1){
  Mtx a(1234, 5);
  init(a);
  Mtx c = a * 3.;
  c.evaluate(arena);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), (ir * c.cols() + ic) * 3.);
}

void matrix_multiply(double* dst, Mtx& a, Mtx& b){
  for (size_t ir = 0; ir < a.rows(); ++ir)
    for (size_t ic = 0; ic < b.cols(); ++ic)
      for (size_t k = 0; k < a.cols(); ++k)
        dst[ir * b.cols() + ic] += a(ir, k) * b(k, ic);
}

TEST_F(ElemOp, MMul1){
  const size_t r1 = 13, r2 = 17, r3 = 27;
  Mtx a(r1, r2), b(r2, r3);
  init(a);
  init2(b);
  Mtx c = a ^ b;
  c.evaluate(arena);

  double expected[r1 * r3];
  memset(expected, 0, r1 * r3 * sizeof(double));
  matrix_multiply(expected, a, b);

  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(expected[ir * c.cols() + ic], c(ir, ic));
}

TEST_F(ElemOp, Transpose1){
  Mtx a(50, 4);
  init2(a);
  Mtx c = ~a;
  c.evaluate(arena);

  EXPECT_EQ(c.rows(), 4);
  EXPECT_EQ(c.cols(), 50);
  for (size_t ir = 0; ir < c.rows(); ++ir)
    for (size_t ic = 0; ic < c.cols(); ++ic)
      EXPECT_DOUBLE_EQ(c(ir, ic), a(ic, ir));
}

TEST_F(ElemOp, Deriviative1){
  Mtx o(128, 128), y(128, 128);
  init2(o);
  init(y);
  Mtx r = ML::isnan0((o - y) / 128);
  r.evaluate(arena);

  for (size_t ir = 0; ir < 128; ++ir)
    for (size_t ic = 0; ic < 128; ++ic)
      EXPECT_DOUBLE_EQ(r(ir, ic), 0.0078125);
}

TEST_F(ElemOp, Deriviative2){
  Mtx o(128, 128), y(128, 128);
  init2(o);
  init_nan(y);
  Mtx r = ML::isnan0((o - y) / 128);
  r.evaluate(arena);

  for (size_t ir = 0; ir < 128; ++ir)
    for (size_t ic = 0; ic < 128; ++ic)
      EXPECT_DOUBLE_EQ(r(ir, ic), 0.);
}

void expected_mse_deriviative(double* r, Mtx& o, Mtx& y){
  for (size_t ir = 0; ir < o.rows(); ++ir)
    for (size_t ic = 0; ic < o.cols(); ++ic){
      double v = (o(ir, ic) - y(ir, ic)) / y.rows();
      if (std::isnan(v))
        r[ir * o.cols() + ic] = 0.;
      else
        r[ir * o.cols() + ic] = v;
    }
}

TEST_F(ElemOp, Deriviative3){
  Mtx o(5000, 3), y(5000, 3);
  init_random(o);
  init_random(y);
  Mtx r = ML::isnan0((o - y) / 5000);
  r.evaluate(arena);

  double* expected = new double[5000 * 3];
  expected_mse_deriviative(expected, o, y);

  for (size_t ir = 0; ir < 5000; ++ir)
    for (size_t ic = 0; ic < 3; ++ic)
      EXPECT_NEAR(r(ir, ic), expected[ir * 3 + ic], 0.00001);
}

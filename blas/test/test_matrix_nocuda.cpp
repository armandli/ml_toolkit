#include <cmath>
#include <vector>
#include <fstream>

#include <gtest/gtest.h>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

TEST(CreateMatrix, CreateMatrix){
  vector<double> v = {1.,2.,3.,4.,5.,6.,7.,8.,9.};

  Mtx a;
  Mtx b(10, 10);
  Mtx c(13, 17, 5.);
  Mtx d(4, 2, v);
  Mtx e = Mtx::zeros(20, 30);
  Mtx f = Mtx::random(31, 22);
  Mtx g = Mtx::ident(4);

  EXPECT_EQ(0, a.rows());
  EXPECT_EQ(0, a.cols());
  EXPECT_DOUBLE_EQ(0., b(1,2));
  EXPECT_DOUBLE_EQ(5., c(10,11));
  EXPECT_DOUBLE_EQ(0., e(15, 7));
  EXPECT_DOUBLE_EQ(1., g(1,1));
  EXPECT_DOUBLE_EQ(0., g(1,2));

  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(v[i * 2 + j], d(i, j));
}

TEST(CopyMatrix, CopyMatrix){
  Mtx a = Mtx::random(13, 21);
  Mtx b = a;
  EXPECT_EQ(a, b);
  Mtx c = move(a);
  EXPECT_EQ(c, b);
}

struct ArithTest : ::testing::Test {
  ArithTest(){
    vector<double> va = {1.,2.,3.,4.};
    vector<double> vb = {2.,3.,4.,5.};
    vector<double> vc = {11.,12.,13.,14.,15.,16.};
    vector<double> vd = {22.,23.,24.,25.,26.,27.};

    a = Mtx(2, 2, va);
    b = Mtx(2, 2, vb);
    c = Mtx(3, 2, vc);
    d = Mtx(2, 3, vd);
    e = Mtx(6, 1, vc);
    f = Mtx(1, 6, vd);
  }

  Mtx a,b,c,d,e,f;
};

TEST_F(ArithTest, Add){
  vector<double> res = {3.,5.,7.,9.};

  Mtx d = a + b;
  a += b;

  EXPECT_EQ(a, d);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], d(i,j));

  vector<double> reg = {4.,5.,6.,7.};
  Mtx c = b + 2.;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(reg[i * 2 + j], c(i, j));
}

TEST_F(ArithTest, Sub){
  vector<double> res = {1.,1.,1.,1.};

  Mtx d = b - a;
  b -= a;

  EXPECT_EQ(b, d);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], d(i, j));

  vector<double> reg = {0.,1.,2.,3.};
  Mtx c = a - 1.;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(reg[i * 2 + j], c(i, j));
}

TEST_F(ArithTest, ElementMul){
  vector<double> res = {2.,4.,6.,8.};
  Mtx d = a * 2.;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], d(i, j));

  vector<double> reg = {1.,1.5,2.,2.5};
  Mtx c = b / 2.;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(reg[i * 2 + j], c(i, j));
}

TEST_F(ArithTest, MatrixMul){
  vector<double> ra = {10., 13., 22., 29.};
  vector<double> rb = {542.,565.,588.,636.,663.,690.,730.,761.,792.};
  vector<double> rc = {242.,253.,264.,275.,286.,297.
                      ,264.,276.,288.,300.,312.,324.
                      ,286.,299.,312.,325.,338.,351.
                      ,308.,322.,336.,350.,364.,378.
                      ,330.,345.,360.,375.,390.,405.
                      ,352.,368.,384.,400.,416.,432.};
  vector<double> rd = {2002.};

  Mtx x = a * b;
  Mtx y = c * d;
  Mtx z = e * f;
  Mtx i = f * e;

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(ra[i * 2 + j], x(i, j));

  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(rb[i * 3 + j], y(i, j));

  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 6; ++j)
      EXPECT_DOUBLE_EQ(rc[i * 6 + j], z(i, j));

  EXPECT_DOUBLE_EQ(rd[0], i(0, 0));
}

TEST_F(ArithTest, ComplexExpression){
  vector<double> res = {-898.,-965.,-1011.,-1087.};
  Mtx k = a + b - d * c;

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], k(i, j));
}

TEST_F(ArithTest, Transpose){
  vector<double> res = {2.,4.,3.,5.};

  Mtx rb = b.transpose();

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], rb(i, j));

  vector<double> reg = {22.,23.,24.,25.,26.,27.};

  Mtx rf = f.transpose();

  for (size_t i = 0; i < 6; ++i)
    EXPECT_DOUBLE_EQ(reg[i], rf(i, 0));

  vector<double> ref = {11.,13.,15.,12.,14.,16.};

  Mtx rc = c.transpose();

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(ref[i * 3 + j], rc(i, j));
}

struct StatTest : ::testing::Test {
  StatTest(){
    vector<double> va = {10., 14., 12., 17., 1., 9.876, 9.877, 17.01, 16.98, 16.99, 15.5, 15.4};
    vector<double> vb = {5.,  4.,  6.,  18., 4., 2.,    3.,    3.,    2.,    9.,    2.,   2.  };
    vector<double> vc = {5.,nan(""),6., 18., 4., 2.,  nan(""), 3.,    2.,    9.,    2.,   nan("")};

    a = Mtx(3, 4, va);
    b = Mtx(4, 3, vb);
    c = Mtx(4, 3, vc);
  }

  Mtx a, b, c;
};

TEST_F(StatTest, MaxCoeff){
  vector<DimV> mra = a.max_coeff(MRow);
  vector<DimV> mca = a.max_coeff(MCol);
  vector<DimV> ma  = a.max_coeff(MAll);

  vector<DimV> mrb = b.max_coeff(MRow);
  vector<DimV> mcb = b.max_coeff(MCol);
  vector<DimV> mb  = b.max_coeff(MAll);

  vector<DimV> mrc = c.max_coeff(MRow);
  vector<DimV> mcc = c.max_coeff(MCol);
  vector<DimV> mc  = c.max_coeff(MAll);

  EXPECT_DOUBLE_EQ(17.01, ma[0].val);
  EXPECT_DOUBLE_EQ(18., mb[0].val);
  EXPECT_DOUBLE_EQ(18., mc[0].val);

  EXPECT_EQ(3, mra[0].idx);
  EXPECT_EQ(3, mra[1].idx);
  EXPECT_EQ(1, mra[2].idx);

  EXPECT_EQ(2, mca[0].idx);
  EXPECT_EQ(2, mca[1].idx);
  EXPECT_EQ(2, mca[2].idx);
  EXPECT_EQ(1, mca[3].idx);

  EXPECT_EQ(2, mrc[0].idx);
  EXPECT_EQ(0, mrc[1].idx);
  EXPECT_EQ(1, mrc[2].idx);
  EXPECT_EQ(0, mrc[3].idx);

  EXPECT_EQ(1, mcc[0].idx);
  EXPECT_EQ(1, mcc[1].idx);
  EXPECT_EQ(0, mcc[2].idx);
}

TEST_F(StatTest, MinCoeff){
  vector<DimV> mra = a.min_coeff(MRow);
  vector<DimV> mca = a.min_coeff(MCol);
  vector<DimV> ma  = a.min_coeff(MAll);

  vector<DimV> mrb = b.min_coeff(MRow);
  vector<DimV> mcb = b.min_coeff(MCol);
  vector<DimV> mb  = b.min_coeff(MAll);

  vector<DimV> mrc = c.min_coeff(MRow);
  vector<DimV> mcc = c.min_coeff(MCol);
  vector<DimV> mc  = c.min_coeff(MAll);

  EXPECT_DOUBLE_EQ(1., ma[0].val);
  EXPECT_DOUBLE_EQ(2., mb[0].val);
  EXPECT_DOUBLE_EQ(2., mc[0].val);

  EXPECT_EQ(0, mra[0].idx);
  EXPECT_EQ(0, mra[1].idx);
  EXPECT_EQ(3, mra[2].idx);

  EXPECT_EQ(1, mca[0].idx);
  EXPECT_EQ(1, mca[1].idx);
  EXPECT_EQ(1, mca[2].idx);
  EXPECT_EQ(2, mca[3].idx);

  EXPECT_EQ(0, mrc[0].idx);
  EXPECT_EQ(2, mrc[1].idx);
  EXPECT_EQ(2, mrc[2].idx);
  EXPECT_EQ(1, mrc[3].idx);

  EXPECT_EQ(0, mcc[0].idx);
  EXPECT_EQ(3, mcc[1].idx);
  EXPECT_EQ(1, mcc[2].idx);
}

TEST_F(StatTest, Sum){
  vector<double> mra = a.sum(MRow);
  vector<double> mca = a.sum(MCol);
  vector<double> ma  = a.sum(MAll);

  vector<double> mrb = b.sum(MRow);
  vector<double> mcb = b.sum(MCol);
  vector<double> mb  = b.sum(MAll);

  vector<double> mrc = c.sum(MRow);
  vector<double> mcc = c.sum(MCol);
  vector<double> mc  = c.sum(MAll);

  EXPECT_DOUBLE_EQ(155.633, ma[0]);
  EXPECT_DOUBLE_EQ(60., mb[0]);
  EXPECT_DOUBLE_EQ(51., mc[0]);
}

TEST_F(StatTest, Mean){
  vector<double> mra = a.mean(MRow);
  vector<double> mca = a.mean(MCol);
  vector<double> ma  = a.mean(MAll);

  vector<double> mrb = b.mean(MRow);
  vector<double> mcb = b.mean(MCol);
  vector<double> mb  = b.mean(MAll);

  vector<double> mrc = c.mean(MRow);
  vector<double> mcc = c.mean(MCol);
  vector<double> mc  = c.mean(MAll);

  EXPECT_DOUBLE_EQ(13.25, mra[0]);
  EXPECT_DOUBLE_EQ(9.44075, mra[1]);
  EXPECT_DOUBLE_EQ(16.2175, mra[2]);

  EXPECT_DOUBLE_EQ(9.326666666666666, mca[0]);
  EXPECT_DOUBLE_EQ(13.622, mca[1]);
  EXPECT_DOUBLE_EQ(12.459, mca[2]);
  EXPECT_DOUBLE_EQ(16.47, mca[3]);

  EXPECT_DOUBLE_EQ(12.969416666666667, ma[0]);

  EXPECT_DOUBLE_EQ(5.5, mrc[0]);
  EXPECT_DOUBLE_EQ(8., mrc[1]);
  EXPECT_DOUBLE_EQ(2.5, mrc[2]);
  EXPECT_DOUBLE_EQ(5.5, mrc[3]);

  EXPECT_DOUBLE_EQ(10.666666666666666, mcc[0]);
  EXPECT_DOUBLE_EQ(3., mcc[1]);
  EXPECT_DOUBLE_EQ(3.3333333333333333, mcc[2]);

  EXPECT_DOUBLE_EQ(5.6666666666666667, mc[0]);
}

TEST(BlockCopyTest, BlockCopyTest){
  vector<double> va = {1.,2.,3.,4.};
  vector<double> vb = {2.,3.,4.,5.};
  vector<double> vc = {3.,4.,5.,6.};

  vector<double> res = {0., 0., 1., 0.,
                        0., 3., 4., 0.,
                        2., 5., 6., 5.,
                        0., 0., 4., 0.};

  Mtx a = Mtx(4, 4, 0.);
  Mtx b = Mtx(1, 4, vb);
  Mtx c = Mtx(4, 1, va);
  Mtx d = Mtx(2, 2, vc);

  a.block(2,0,1,4) = b;
  a.block(0,2,4,1) = c;
  a.block(1,1,2,2) = d;

  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
      EXPECT_DOUBLE_EQ(res[i * 4 + j], a(i, j));
}

TEST(SaveLoadTest, SaveLoadTest){
  Mtx m = Mtx::random(16, 16);

  fstream fout("test_matrix_save.sav", fstream::out );
  m.save(fout);

  fstream fin("test_matrix_save.sav", fstream::in );
  Mtx n(fin);

  EXPECT_EQ(m, n);
}

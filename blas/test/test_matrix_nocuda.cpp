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

  Mtx x = a * b;
  Mtx y = c * d;
  Mtx z = e * f;

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(ra[i * 2 + j], x(i, j));

  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(rb[i * 3 + j], y(i, j));

  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 6; ++j)
      EXPECT_DOUBLE_EQ(rc[i * 6 + j], z(i, j));
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

  b.transpose();

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(res[i * 2 + j], b(i, j));

  vector<double> reg = {22.,23.,24.,25.,26.,27.};

  f.transpose();

  for (size_t i = 0; i < 6; ++i)
    EXPECT_DOUBLE_EQ(reg[i], f(i, 0));

  vector<double> ref = {11.,13.,15.,12.,14.,16.};

  cout << c << endl;

  c.transpose();

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(ref[i * 3 + j], c(i, j));
  
  cout << c << endl;
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

//  fstream fin("test_matrix_save.sav", fstream::in );
//  Mtx n(fin);

//  EXPECT_EQ(m, n);
}

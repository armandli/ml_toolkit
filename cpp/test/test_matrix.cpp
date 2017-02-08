#include <matrix.h>

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <cmath>

using namespace std;

using tt = double;

TEST(CD, Construction){
  Mtx<tt> def;

  Mtx<tt> cm(2, 3, 10.);
  
  EXPECT_EQ(2, cm.rows());
  EXPECT_EQ(3, cm.cols());
  for (size_t i = 0; i < cm.rows(); ++i)
    for (size_t j = 0; j < cm.cols(); ++j)
      EXPECT_EQ(10., cm(i, j));

  vector<tt> v = {1., 2., 3., 4., 5., 6., 7., 8.};
  Mtx<tt> cv(2, 4, v);

  EXPECT_EQ(2, cv.rows());
  EXPECT_EQ(4, cv.cols());
  for (size_t i = 0; i < cv.rows(); ++i)
    for (size_t j = 0; j < cv.cols(); ++j)
      EXPECT_EQ(v[i + j * 2], cv(i, j));
}

TEST(CD, CopyConstruction){
  vector<tt> v = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  Mtx<tt> a(3, 3, v);

  Mtx<tt> b(a);

  EXPECT_EQ(3, b.rows());
  EXPECT_EQ(3, b.cols());
  for (size_t i = 0; i < b.rows(); ++i)
    for (size_t j = 0; j < b.cols(); ++j)
      EXPECT_EQ(v[i + j * 3], b(i, j));
}

TEST(CD, MoveConstruction){
  vector<tt> v = {1., 2., 3., 4.};

  Mtx<tt>* a = new Mtx<tt>(4, 1, v);
  Mtx<tt> b(std::move(*a));

  EXPECT_EQ(4, b.rows());
  EXPECT_EQ(1, b.cols());
  for (size_t i = 0; i < b.rows(); ++i)
    for (size_t j = 0; j < b.cols(); ++j)
      EXPECT_EQ(v[i + j * 4], b(i, j));
}

TEST(CD, CopyAssignment){
  vector<tt> v = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.};
  Mtx<tt> a(4, 4, v);
  Mtx<tt> c(2, 2, 0.); c = a;

  EXPECT_EQ(4, c.rows());
  EXPECT_EQ(4, c.cols());
  for (size_t i = 0; i < c.rows(); ++i)
    for (size_t j = 0; j < c.cols(); ++j)
      EXPECT_EQ(v[i + j * 4], c(i, j));

  //test assign to self
  Mtx<tt> d(1, 10, 9.); d = d;
  for (size_t i = 0; i < d.rows(); ++i)
    for (size_t j = 0; j < d.cols(); ++j)
      EXPECT_EQ(9., d(i, j));
}

TEST(CD, MoveAssignment){
  vector<tt> v = {1., 2., 3., 4., 5., 6.};

  Mtx<tt> b(2, 3, 16.);
  Mtx<tt>* a = new Mtx<tt>(3, 2, v);
  b = std::move(*a);

  EXPECT_EQ(3, b.rows());
  EXPECT_EQ(2, b.cols());
  for (size_t i = 0; i < b.rows(); ++i)
    for (size_t j = 0; j < b.cols(); ++j)
      EXPECT_EQ(v[i + j * 3], b(i, j));
}

TEST(Transpose, Transpose){
  vector<tt> v1 = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  vector<tt> v2 = {1., 2., 3., 4., 5., 6.};
  vector<tt> v3 = {1., 4., 2., 6., 5.};

  Mtx<tt> a(3, 3, v1); a.flip();
  Mtx<tt> b(2, 3, v2); b.flip();
  Mtx<tt> c(3, 2, v2); c.flip();
  Mtx<tt> d(5, 1, v3); d.flip();
  Mtx<tt> e(1, 5, v3); e.flip();

  for (size_t i = 0; i < a.rows(); ++i)
    for (size_t j = 0; j < a.cols(); ++j)
      EXPECT_EQ(v1[i + j * 3], a(i, j));

  for (size_t i = 0; i < b.rows(); ++i)
    for (size_t j = 0; j < b.cols(); ++j)
      EXPECT_EQ(v2[i + j * 2], b(i, j));

  for (size_t i = 0; i < c.rows(); ++i)
    for (size_t j = 0; j < c.cols(); ++j)
      EXPECT_EQ(v2[i + j * 3], c(i, j));

  for (size_t i = 0; i < d.rows(); ++i)
    for (size_t j = 0; j < d.cols(); ++j)
      EXPECT_EQ(v3[i + j * 5], d(i, j));

  for (size_t i = 0; i < e.rows(); ++i)
    for (size_t j = 0; j < e.cols(); ++j)
      EXPECT_EQ(v3[i + j * 1], e(i, j));
}

TEST(SaveLoadMatrix, SaveLoadMatrix){
  Mtx<tt> a(10, 10, 0.);
  D2IterC(ir, ic, 0, a.rows(), 0, a.cols()) a(ir, ic) = ir * ic + 1.;
  a.save("tmp.mtx");

  Mtx<tt> b(string("tmp.mtx"));

  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  D2IterC(ir, ic, 0, a.rows(), 0, a.cols()) EXPECT_DOUBLE_EQ(a(ir, ic), b(ir, ic));
}

struct ArithTest : ::testing::Test {
  ArithTest(){
    vector<tt> va = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    vector<tt> vb = {1., 2., 3., 4., 5., 6.};
    vector<tt> vc = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    vector<tt> vd = {1., 2., 3.};

    vector<tt> ve(1000 * 1000);
    for (size_t i = 0; i < 1000 * 1000; ++i)
      ve[i] = (double)rand() / 10.;

    vector<tt> vf(100 * 100);
    for (size_t i = 0; i < 100 * 100; ++i)
      vf[i] = (double)rand() / 10.;

    a = Mtx<tt>(3, 3, va);
    b = Mtx<tt>(2, 3, vb);
    c = Mtx<tt>(3, 2, vb);
    d = Mtx<tt>(3, 4, vc);
    e = Mtx<tt>(4, 3, vc);
    f = Mtx<tt>(1, 3, vd);
    g = Mtx<tt>(3, 1, vd);
    h = Mtx<tt>(1000, 1000, ve);
    i = Mtx<tt>(1000, 1000, ve);
    j = Mtx<tt>(100, 100, vf);
    k = Mtx<tt>(100, 100, vf);
  }

  Mtx<tt> a, b, c, d, e, f, g, h, i, j, k;
};

TEST_F(ArithTest, Transpose){
  Mtx<tt> ta = a.transpose();
  vector<tt> expected1 = {1., 4., 7., 2., 5., 8., 3., 6., 9.};
  EXPECT_EQ(3, ta.rows());
  EXPECT_EQ(3, ta.cols());
  D2IterC(ir, ic, 0, ta.rows(), 0, ta.cols()) EXPECT_DOUBLE_EQ(expected1[ir + ic * ta.rows()], ta(ir, ic));

  Mtx<tt> te = e.transpose();
  vector<tt> expected2 = {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.};
  EXPECT_EQ(3, te.rows());
  EXPECT_EQ(4, te.cols());
  D2IterC(ir, ic, 0, te.rows(), 0, te.cols()) EXPECT_DOUBLE_EQ(expected2[ir + ic * te.rows()], te(ir, ic));

  Mtx<tt> tg = g.transpose();
  vector<tt> expected3 = {1., 2., 3.};
  EXPECT_EQ(1, tg.rows());
  EXPECT_EQ(3, tg.cols());
  D2IterC(ir, ic, 0, tg.rows(), 0, tg.cols()) EXPECT_DOUBLE_EQ(expected3[ir + ic * tg.rows()], tg(ir, ic));

  a.t();
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(3, a.cols());
  D2IterC(ir, ic, 0, a.rows(), 0, a.cols()) EXPECT_DOUBLE_EQ(expected1[ir + ic * a.rows()], a(ir, ic));

  e.t();
  EXPECT_EQ(3, e.rows());
  EXPECT_EQ(4, e.cols());
  D2IterC(ir, ic, 0, e.rows(), 0, e.cols()) EXPECT_DOUBLE_EQ(expected2[ir + ic * e.rows()], e(ir, ic));

  g.t();
  EXPECT_EQ(1, g.rows());
  EXPECT_EQ(3, g.cols());
  D2IterC(ir, ic, 0, g.rows(), 0, g.cols()) EXPECT_DOUBLE_EQ(expected3[ir + ic * g.rows()], g(ir, ic));
}

TEST_F(ArithTest, AddScalar){
  tt val = 13.;
  vector<tt> expected = {14., 15., 16., 17., 18., 19., 20., 21., 22.};

  a.add(val);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, SubScalar){
  tt val = 10.;
  vector<tt> expected = {-9., -8., -7., -6., -5., -4., -3., -2., -1};

  a.sub(val);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, MulScalar){
  tt val = 2.;
  vector<tt> expected = {2., 4., 6., 8., 10., 12., 14., 16., 18.};

  a.mul(val);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, DivScalar){
  tt val = 0.5;
  vector<tt> expected = {2., 4., 6., 8., 10., 12., 14., 16., 18.};

  a.div(val);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, PowScalar){
  tt val = 2.;
  vector<tt> expected = {1., 4., 9., 16., 25., 36., 49., 64., 81.};

  a.pow(val);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, FuncSclar){
  vector<tt> expected = {0.76159415595576485, 0.9640275800758169, 0.99505475368673046, 0.99932929973906703, 0.99990920426259511, 0.99998771165079559, 0.99999833694394469, 0.99999977492967584, 0.999999969540041};

  a.foreach([](tt& v){ v = tanh(v); });
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}

TEST_F(ArithTest, AddMtx){
  vector<tt> expected = {2., 4., 6., 8., 10., 12., 14., 16., 18.};

  a.add(a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, SubMtx){
  vector<tt> expected = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

  a.sub(a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, MulMtx){
  vector<tt> expected = {1., 4., 9., 16., 25., 36., 49., 64., 81.};

  a.mul(a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, DivMtx){
  vector<tt> expected = {1., 1., 1., 1., 1., 1., 1., 1., 1.};

  a.div(a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, PowMtx){
  vector<tt> expected = {1.0, 4.0, 27.0, 256.0, 3125.0, 46656.0, 823543.0, 16777216.0, 387420489.0};

  a.pow(a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}
TEST_F(ArithTest, FuncMtx){
  vector<tt> expected = {1., 4., 9., 16., 25., 36., 49., 64., 81.};

  a.foreach([](tt& a, const tt& b){
    a = a * b;
  }, a);
  D2IterC(ir, ic, 0, a.rows(), 0., a.cols()) EXPECT_DOUBLE_EQ(expected[ir + ic * a.rows()], a(ir, ic));
}

TEST_F(ArithTest, DotMtx){
  vector<tt> expected1 = {30., 36., 42., 66., 81., 96., 102., 126., 150.};
  Mtx<tt> ra = a * a;
  D2IterC(ir, ic, 0, ra.rows(), 0, ra.cols()) EXPECT_DOUBLE_EQ(expected1[ir + ic * ra.rows()], ra(ir, ic));

  vector<tt> expected2 = {22., 28., 49., 64.};
  Mtx<tt> rb = b * c;
  EXPECT_EQ(2, rb.rows());
  EXPECT_EQ(2, rb.cols());
  D2IterC(ir, ic, 0, rb.rows(), 0, rb.cols()) EXPECT_DOUBLE_EQ(expected2[ir + ic * rb.rows()], rb(ir, ic));

  vector<tt> expected3 = {1., 2. ,3., 2., 4., 6., 3., 6., 9.};
  Mtx<tt> rc = g * f;
  EXPECT_EQ(3, rc.rows());
  EXPECT_EQ(3, rc.cols());
  D2IterC(ir, ic, 0, rc.rows(), 0, rc.cols()) EXPECT_DOUBLE_EQ(expected3[ir + ic * rc.rows()], rc(ir, ic));

  //flip test
  vector<tt> expected4 = {9., 12., 15., 19., 26., 33., 29., 40., 51.};
  Mtx<tt> rd = c.flip() * b;
  EXPECT_EQ(3, rd.rows());
  EXPECT_EQ(3, rd.cols());
  D2IterC(ir, ic, 0, rd.rows(), 0, rd.cols()) EXPECT_DOUBLE_EQ(expected4[ir + ic * rd.rows()], rd(ir, ic));

  vector<tt> expected5 = {70., 80., 90., 158., 184., 210., 246., 288., 330.};
  Mtx<tt> re = d.flip() * e;
  EXPECT_EQ(3, re.rows());
  EXPECT_EQ(3, re.cols());
  D2IterC(ir, ic, 0, re.rows(), 0, re.cols()) EXPECT_DOUBLE_EQ(expected5[ir + ic * re.rows()], re(ir, ic));

  vector<tt> expected6 = {14.};
  Mtx<tt> rf = f.flip() * g;
  EXPECT_EQ(1, rf.rows());
  EXPECT_EQ(1, rf.cols());
  D2IterC(ir, ic, 0, rf.rows(), 0, rf.cols()) EXPECT_DOUBLE_EQ(expected6[ir + ic * rf.rows()], rf(ir, ic));

  Mtx<tt> rg = g.flip() * f;
  EXPECT_EQ(3, rg.rows());
  EXPECT_EQ(3, rg.cols());
  D2IterC(ir, ic, 0, rg.rows(), 0, rg.cols()) EXPECT_DOUBLE_EQ(expected3[ir + ic * rg.rows()], rg(ir, ic));

  //performance results
  clock_t start;
  start = clock();
  Mtx<tt> rp1 = h * i;
  std::cout << "No Perf 1000 * 1000: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  start = clock();
  Mtx<tt> rp2 = h.flip() * i;
  std::cout << "Perf 1000 * 1000: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  start = clock();
  Mtx<tt> rp3 = j * k;
  std::cout << "No Perf 100 * 100: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  start = clock();
  Mtx<tt> rp4 = j.flip() * k;
  std::cout << "Perf 100 * 100: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}

TEST_F(ArithTest, Sum){
  vector<tt> sum1 = d.sum(MRow);
  vector<tt> expected1 = {22., 26., 30.};
  D1Iter(i, 0, expected1.size()) EXPECT_DOUBLE_EQ(expected1[i], sum1[i]);

  vector<tt> sum2 = d.sum(MCol);
  vector<tt> expected2 = {6., 15., 24., 33.};
  D1Iter(i, 0, expected2.size()) EXPECT_DOUBLE_EQ(expected2[i], sum2[i]);

  vector<tt> sum3 = f.sum(MCol);
  vector<tt> expected3 = {1., 2., 3.};
  D1Iter(i, 0, expected3.size()) EXPECT_DOUBLE_EQ(expected3[i], sum3[i]);

  vector<tt> sum4 = f.sum(MRow);
  vector<tt> expected4 = {6.};
  D1Iter(i, 0, expected4.size()) EXPECT_DOUBLE_EQ(expected4[i], sum4[i]);
}

TEST_F(ArithTest, Mean){
  vector<tt> sum1 = d.mean(MRow);
  vector<tt> expected1 = {5.5, 6.5, 7.5};
  D1Iter(i, 0, expected1.size()) EXPECT_DOUBLE_EQ(expected1[i], sum1[i]);

  vector<tt> sum2 = d.mean(MCol);
  vector<tt> expected2 = {2., 5., 8., 11.};
  D1Iter(i, 0, expected2.size()) EXPECT_DOUBLE_EQ(expected2[i], sum2[i]);

  vector<tt> sum3 = f.mean(MCol);
  vector<tt> expected3 = {1., 2., 3.};
  D1Iter(i, 0, expected3.size()) EXPECT_DOUBLE_EQ(expected3[i], sum3[i]);

  vector<tt> sum4 = f.mean(MRow);
  vector<tt> expected4 = {2.};
  D1Iter(i, 0, expected4.size()) EXPECT_DOUBLE_EQ(expected4[i], sum4[i]);
}

TEST_F(ArithTest, Maxi){
  vector<size_t> va = a.maxi(MCol);
  vector<size_t> expected1 = {2, 2, 2};
  D1Iter(i, 0, a.cols()) EXPECT_EQ(expected1[i], va[i]);

  vector<size_t> vb = a.maxi(MRow);
  vector<size_t> expected2 = {2, 2, 2};
  D1Iter(i, 0, a.rows()) EXPECT_EQ(expected2[i], vb[i]);
}

TEST_F(ArithTest, Mini){
  vector<size_t> va = a.mini(MCol);
  vector<size_t> expected1 = {0, 0, 0};
  D1Iter(i, 0, a.cols()) EXPECT_EQ(expected1[i], va[i]);

  vector<size_t> vb = a.mini(MRow);
  vector<size_t> expected2 = {0, 0, 0};
  D1Iter(i, 0, a.rows()) EXPECT_EQ(expected2[i], vb[i]);
}


TEST_F(ArithTest, RowRef){
  vector<tt> expected1 = {3., 6., 9.};
  RowRef<tt> ar = a.row(2);
  D1Iter(i, 0, a.cols()) EXPECT_DOUBLE_EQ(expected1[i], ar[i]);

  vector<tt> expected2 = {2., 4., 6.};
  RowRef<tt> br = b.row(1);
  D1Iter(i, 0, b.cols()) EXPECT_DOUBLE_EQ(expected2[i], br[i]);

  vector<tt> expected3 = {2., 6., 10.};
  RowRef<tt> er = e.row(1);
  D1Iter(i, 0, e.cols()) EXPECT_DOUBLE_EQ(expected3[i], er[i]);

  vector<tt> expected4 = {1., 2., 3.};
  RowRef<tt> fr = f.row(0);
  D1Iter(i, 0, f.cols()) EXPECT_DOUBLE_EQ(expected4[i], fr[i]);
}

TEST_F(ArithTest, ColRef){
  vector<tt> expected1 = {4., 5., 6.};
  ColRef<tt> ac = a.col(1);
  D1Iter(i, 0, a.rows()) EXPECT_DOUBLE_EQ(expected1[i], ac[i]);

  vector<tt> expected2 = {1., 2., 3.};
  ColRef<tt> cc = c.col(0);
  D1Iter(i, 0, c.rows()) EXPECT_DOUBLE_EQ(expected2[i], cc[i]);

  vector<tt> expected3 = {5., 6., 7., 8.};
  ColRef<tt> ec = e.col(1);
  D1Iter(i, 0, e.rows()) EXPECT_DOUBLE_EQ(expected3[i], ec[i]);

  vector<tt> expected4 = {1., 2., 3.};
  ColRef<tt> gc = g.col(0);
  D1Iter(i, 0, g.rows()) EXPECT_DOUBLE_EQ(expected4[i], gc[i]);
}

TEST_F(ArithTest, RowRefForeach){
  for (size_t i = 0; i < a.rows(); ++i){
    RowRef<tt> r = a.row(i);
    r.foreach([i](tt& d){
        d = d * i;
    });
  }
  vector<tt> expected1 = {0., 2., 6., 0., 5., 12., 0., 8., 18.};
  D2IterC(ir, ic, 0, a.rows(), 0, a.cols()) EXPECT_DOUBLE_EQ(expected1[ir + ic * a.rows()], a(ir, ic));
}

TEST_F(ArithTest, ColRefForeach){
  for (size_t i = 0; i < e.cols(); ++i){
    ColRef<tt> c = e.col(i);
    c.foreach([i](tt& d){
        d = d * i;
    });
  }
  vector<tt> expected1 = {0., 0., 0., 0., 5., 6., 7., 8., 18., 20., 22., 24.};
  D2IterC(ir, ic, 0, e.rows(), 0, e.cols()) EXPECT_DOUBLE_EQ(expected1[ir + ic * e.rows()], e(ir, ic));
}

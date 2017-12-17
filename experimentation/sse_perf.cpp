#include <Eigen/Dense>
#include <ml_sse.h>
#include <ctime>
#include <iostream>
using namespace std;
using Eigen::MatrixXd;

void foo(double* a, size_t rowstride, size_t colstride){
  ML::SSE::emul_const_2d_sse_pd(a, a, -1., rowstride, colstride, colstride);
  ML::SSE::exp_2d_sse_pd(a, a, rowstride, colstride, colstride);
  ML::SSE::add_const_2d_sse_pd(a, a, 1., rowstride, colstride, colstride);
  ML::SSE::ediv_cm_2d_sse_pd(1., a, a, rowstride, colstride, colstride);
}

void goo(double* a, double* b, size_t rowstride, size_t colstride){
  ML::SSE::sigmoid_2d_sse_pd(b, a, rowstride, colstride, colstride);
}

int main(){
  double* a = new double[MTX_BLOCK_RSZ * 1000 * 1000], * b = new double[MTX_BLOCK_RSZ * 1000 * 1000];
  MatrixXd c = MatrixXd::Random(1000, 4000);

  clock_t start = clock();

  for (size_t i = 0; i < 1000; ++i)
    foo(a, 1000, 4000);

  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  start = clock();

  for (size_t i = 0; i < 1000; ++i)
    goo(a, b, 1000, 4000);

  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  start = clock();

  double tmp;
  for (size_t i = 0; i < 100; ++i){
    MatrixXd d = c.unaryExpr([](double d){ return 1. / (1. + exp(d * -1.)); });
    tmp = d(0, 0);
  }

  cout << tmp << endl;
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

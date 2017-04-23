#include <cblas.h>
#include <cstdio>

#include <cassert>
#include <ctime>
#include <random>
#include <iostream>

#define SZ (128 * 16)

using namespace std;

//dgemm routine calculates:
//C = alpha * A * B + beta * C
//dgemm routine parameter explanation:
//  cblas_dgemm(
//    CblasRowMajor, - indicate matrices are stored in row major order
//    CblasNoTrans,  - indicate A should not be transposed before multiplication
//    CblasNoTrans,  - indicate B should not be transposed before multiplication
//    m, n, k,       - indicate size of matrices, A = m rows by k columns, B = k rows by n columns, C = m rows by n columns
//    alpha,         - multiplier on A
//    A,             - array used to store A
//    k,             - leading dimension of array A, for row major this is same as number of columns
//    B,             - array used to store B
//    n,             - leading dimension of array B, or number of elements between successive rows (row major)
//    beta,          - beta multiplier on C
//    C,             - array used to store C
//    n,             - leading dimension of array C
//  )

struct Mtx {
  double* data;
  size_t rows;
  size_t cols;
  
  Mtx(size_t rows, size_t cols):
    data(new double[rows * cols]), rows(rows), cols(cols)
  {}
  ~Mtx(){
    delete[] data;
  }
};

ostream& operator<<(ostream& out, Mtx& m){
  for (size_t i = 0; i < m.rows; ++i){
    for (size_t j = 0; j < m.cols; ++j)
      cout << m.data[i * m.cols + j] << " ";
    cout << endl;
  }
  return out;
}

default_random_engine& get_default_random_engine(){
  static default_random_engine eng(time(0));
  return eng;
}

void random_matrix(Mtx& m){
  uniform_real_distribution<double> dist(-100.F, 100.F);
  default_random_engine& eng = get_default_random_engine();

  for (size_t i = 0; i < m.rows * m.cols; ++i)
    m.data[i] = dist(eng);
}

void generate_sample(Mtx& a, Mtx& b){
  for (size_t i = 0; i < a.rows * a.cols; ++i)
    a.data[i] = (double)i;
  for (size_t i = 0; i < b.rows * b.cols; ++i)
    b.data[i] = (double)i;
}

clock_t matrix_multiply(Mtx& c, Mtx& a, Mtx& b){
  for (size_t i = 0; i < c.rows; ++i)
    for (size_t j = 0; j < c.cols; ++j){
      c.data[i * c.cols + j] = 0.;
      for (size_t k = 0; k < a.cols; ++k)
        c.data[i * c.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
    }

  return clock();
}

void matrix_multiply_cblas(Mtx& c, Mtx& a, Mtx& b){
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              a.rows, b.cols, a.cols, 1.,
              a.data, a.cols,
              b.data, b.cols,
              0., c.data, c.cols
             );
}

void simple_test(){
  double A[6] = {1.0,2.0,1.0,
                -3.0,4.0,-1.0};         
  double B[6] = {1.0,2.0,
                 1.0,-3.0,
                 4.0,-1.0};  
  double C[9] = {.5,.5,.5,
                 .5,.5,.5,
                 .5,.5,.5}; 
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,3,3,2,1.,A, 3, B, 2,0.,C,3);

  //print as row major result
  for(int i=0; i < 3; i++){
    for (int j = 0; j < 3; ++j)
      printf("%lf ", C[j * 3 + i]);
    printf("\n");
  }
}

int main(){
  Mtx a(SZ, SZ), b(SZ, SZ), c(SZ, SZ), d(SZ, SZ);

  generate_sample(a, b);
  for (size_t i = 0; i < c.rows * c.cols; ++i)
    c.data[i] = 0.;

  clock_t timing_start = clock();
  matrix_multiply_cblas(c, a, b);
  cout << "Time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  matrix_multiply(d, a, b);

  bool is_same = true;
  for (size_t i = 0; i < c.rows; ++i)
    for (size_t j = 0; j < c.cols; ++j)
      if (c.data[i * c.cols + j] != d.data[i * d.cols + j]){
        cout << "Result Unequal by " << (c.data[i * c.cols + j] - d.data[i * d.cols + j]) << endl;
        is_same = false;
        goto m;
      }
m:if (is_same) cout << "Result equal" << endl;

  simple_test();
}

#include <curand.h>
#include <cublas_v2.h>

#include <cassert>
#include <ctime>
#include <random>
#include <iostream>

#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {   
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }   
}

#define BSZ 128
#define TSZ 16
#define SZ (BSZ * TSZ)
#define SZA 2000

using namespace std;

struct Mtx {
  double* data;
  size_t rows;
  size_t cols;
  size_t rowstride;
  size_t colstride;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols, size_t rowstride, size_t colstride):
    data(nullptr), rows(rows), cols(cols), rowstride(rowstride), colstride(colstride), is_cuda(is_cuda) {
    if (is_cuda) { gpu_errchk(cudaMalloc(&data, sizeof(double) * rowstride * colstride)); }
    else {
      data = new double[rowstride * colstride];
      memset(data, 0, sizeof(double) * rowstride * colstride);
    }
  }
  ~Mtx(){
    if (is_cuda) { gpu_errchk(cudaFree(data)); }
    else         delete[] data;
  }
};

ostream& operator<<(ostream& out, Mtx& m){
  assert(m.is_cuda == false);

  for (size_t i = 0; i < m.rows; ++i){
    for (size_t j = 0; j < m.cols; ++j)
      cout << m.data[i * m.colstride + j] << " ";
    cout << endl;
  }
  return out;
}

//only can generate random floats
//void random_matrix_cuda(Mtx& m){
//  assert(m.is_cuda);
//
//  curandGenerator_t prng;
//  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//
//  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//  curandGenerateUniform(prng, m.data, m.rows * m.cols);
//}

default_random_engine& get_default_random_engine(){
  static default_random_engine eng(time(0));
  return eng;
}

void random_matrix(Mtx& m){
  assert(m.is_cuda == false);

  uniform_real_distribution<double> dist(-100.F, 100.F);
  default_random_engine& eng = get_default_random_engine();

  for (size_t i = 0; i < m.rows * m.cols; ++i)
    m.data[i] = dist(eng);
}

void generate_sample(Mtx& a, Mtx& b){
  assert(a.is_cuda == false && b.is_cuda == false);
  for (size_t ir = 0; ir < a.rows; ++ir)
    for (size_t ic = 0; ic < a.cols; ++ic)
      a.data[ir * a.colstride + ic] = (double)1.;
  for (size_t ir = 0; ir < b.rows; ++ir)
    for (size_t ic = 0; ic < b.cols; ++ic)
      b.data[ir * b.colstride + ic] = (double)1.;
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

void mmul_cublas(Mtx& c, Mtx& a, Mtx& b){
  int lda = a.colstride, ldb = b.colstride, ldc = b.colstride;
  const double alpha = 1.;
  const double beta = 0.;
  const double* palpha = &alpha;
  const double* pbeta = &beta;

  //recommended to save the handle for multiple use
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows, b.cols, a.cols, palpha, a.data, lda, b.data, ldb, pbeta, c.data, ldc);

  cublasDestroy(handle);
}

int main(){
  Mtx a(false, SZA, SZA, SZ, SZ), b(false, SZA, SZA, SZ, SZ), c(false, SZA, SZA, SZ, SZ), d(false, SZA, SZA, SZ, SZ);
  Mtx da(true, SZA, SZA, SZ, SZ), db(true, SZA, SZA, SZ, SZ), dc(true, SZA, SZA, SZ, SZ);

//  Mtx a(false, SZ, SZ, SZ, SZ), b(false, SZ, SZ, SZ, SZ), c(false, SZ, SZ, SZ, SZ), d(false, SZ, SZ, SZ, SZ);
//  Mtx da(true, SZ, SZ, SZ, SZ), db(true, SZ, SZ, SZ, SZ), dc(true, SZ, SZ, SZ, SZ);

  generate_sample(a, b);

//  random_matrix(a);
//  random_matrix(b);

  clock_t timing_start = clock();
  cudaMemcpy(da.data, a.data, sizeof(double) * a.rowstride * a.colstride, cudaMemcpyHostToDevice);
  cudaMemcpy(db.data, b.data, sizeof(double) * b.rowstride * b.colstride, cudaMemcpyHostToDevice);

  mmul_cublas(dc, da, db);

  cudaMemcpy(c.data, dc.data, sizeof(double) * c.rowstride * c.colstride, cudaMemcpyDeviceToHost);
  cout << "Time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cout << c;

//  timing_start = clock();
//  clock_t timing_end = matrix_multiply(d, a, b);

//  bool is_same = true;
//  for (size_t i = 0; i < c.rows; ++i)
//    for (size_t j = 0; j < c.cols; ++j)
//      if (c.data[i * c.cols + j] != d.data[i * d.cols + j]){
//        cout << "Result Unequal by " << (c.data[i * c.cols + j] - d.data[i * d.cols + j]) << endl;
//        is_same = false;
//        break;
//      }
//  if (is_same) cout << "Result equal" << endl;
}

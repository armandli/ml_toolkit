#include <curand.h>
#include <cublas_v2.h>

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

#define BSZ 64
#define TSZ 16
#define SZ (BSZ * TSZ)

using namespace std;

struct Mtx {
  double* data;
  size_t rows;
  size_t cols;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols):
    data(nullptr), rows(rows), cols(cols), is_cuda(is_cuda) {
    if (is_cuda) { gpu_errchk(cudaMalloc(&data, sizeof(double) * rows, cols)); }
    else         data = new double[rows * cols];
  }
  ~Mtx(){
    if (is_cuda) { gnu_errchk(cudaFree(data)); }
    else         delete[] data;
  }
};

void random_matrix_cuda(Mtx& m){
  assert(m.is_cuda);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  curandGenerateUniform(prng, m.data, m.rows * m.cols);
}

default_random_engine& get_default_random_engine(){
  static default_random_engine eng(time(0));
  return eng;
}

void random_matrix(Mtx& m){
  assert(m.is_cuda == false);

  uniform_real_distribution<T> dist(-100.F, 100.F);
  default_random_engine& eng = get_default_random_engine();

  for (size_t i = 0; i < sz; ++i)
    m.data[i] = dist(eng);
}

void generate_sample(Mtx& a, Mtx& b){
  //TODO
}

void mmul_cublas(Mtx& c, Mtx& a, Mtx& b){
  int lda = a.rows, ldb = a.cols, ldc = a.rows;
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
  Mtx a(false, SZ, SZ), b(false, SZ, SZ), c(false, SZ, SZ), d(false, SZ, SZ);
  Mtx da(true, SZ, SZ), db(true, SZ, SZ), dc(true, SZ, SZ);

  random_matrix(a);
  random_matrix(b);

  cudaMemcpy(da.data, a, sizeof(double) * a.rows * a.cols, cudaMemcpyHostToDevice);
  cudaMemcpy(db.data, b, sizeof(double) * b.rows * b.cols, cudaMemcpyHostToDevice);

  mmul_cublas(dc, da, db);

  cudaMemcpy(c.data, dc.data, sizeof(double) * c.rows * c.cols, cudaMemcpyDeviceToHost);
}

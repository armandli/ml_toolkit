#include <cassert>
#include <cstdio>
#include <ctime>
#include <random>
#include <iostream>

//#define BSZ 128
//#define TSZ 16
//#define SZ (BSZ * TSZ)

#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {   
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }   
}

using namespace std;

template <typename T>
struct Mtx {
  T* data;
  size_t rows;
  size_t cols;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols):
    data(0), rows(rows), cols(cols), is_cuda(is_cuda) {
    if (is_cuda) { gpu_errchk(cudaMalloc(&data, sizeof(T) * rows * cols)); }
    else         data = new T[rows * cols];
  }
  ~Mtx(){
    if (is_cuda) { gpu_errchk(cudaFree(data)); }
    else         delete[] data;
  }
};

template <typename T>
ostream& operator<<(ostream& out, Mtx<T>& m){
  assert(m.is_cuda == false);

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

template <typename T>
void random_matrix(Mtx<T>& m){
  assert(m.is_cuda == false);

  uniform_real_distribution<T> dist(-100.F, 100.F);
  default_random_engine& eng = get_default_random_engine();

  for (size_t i = 0; i < m.rows * m.cols; ++i)
    m.data[i] = dist(eng);
}

template <typename T>
void unary_init(Mtx<T>& a){
  for (size_t i = 0; i < a.rows; ++i)
    for (size_t j = 0; j < a.cols; ++j)
      a.data[i * a.cols + j] = (i * a.cols + j);
}

template <typename T>
void binary_init(Mtx<T>& a, Mtx<T>& b){
  for (size_t i = 0; i < a.rows; ++i)
    for (size_t j = 0; j < a.cols; ++j)
      a.data[i * a.cols + j] = b.data[i * a.cols + j] = (double)(i * a.cols + j);
}


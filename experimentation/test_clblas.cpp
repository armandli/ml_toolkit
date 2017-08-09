#include <cassert>
#include <ctime>
#include <vector>
#include <string>
#include <random>
#include <utility>
#include <iostream>

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <clBLAS.h>
#include <clblast.h>

#define BSZ 126
#define TSZ 16
#define SZ (BSZ * TSZ)

using namespace std;
using namespace clblast;

struct Mtx {
  double* data;
  size_t rows;
  size_t cols;

  Mtx(size_t rows, size_t cols):
    data(new double[rows * cols]), rows(rows), cols(cols) {
  }
  ~Mtx(){
    if (data){
      delete[] data;
      data = nullptr;
    }
  }
};

ostream& operator<<(ostream& out, Mtx& m){
  for (size_t i = 0; i < m.rows; ++i){
    for (size_t j = 0; j < m.cols; ++j)
      out << m.data[i * m.cols + j] << " ";
    out << endl;
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

void generate_sample(Mtx& a, Mtx& b, Mtx& c){
  for (size_t i = 0; i < a.rows * a.cols; ++i)
    a.data[i] = (double)i;
  for (size_t i = 0; i < b.rows * b.cols; ++i)
    b.data[i] = (double)i;
  for (size_t i = 0; i < c.rows * c.cols; ++i)
    c.data[i] = 0.;
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

template <typename T>
class optional {
  union Storage {
    unsigned char dummy;
    T t;

    Storage(): dummy() {}
    Storage(const T& t): t(t) {}

    Storage(const Storage& o): t(o.t) {}
    Storage& operator=(const Storage& o){
      t = o.t;
      return *this;
    }

    ~Storage(){}

  } v;

  bool isNone;

public:
  optional(): v(), isNone(true) {}
  explicit optional(const T& t): v(t), isNone(false) {}

  ~optional(){
    if (isNone == false)
      v.t.T::~T();
  }

  const T& get() const {
    assert(isNone == false);
    return v.t;
  }
  T& get(){
    assert(isNone == false);
    return v.t;
  }

  void set(T& t){
    v.t = t;
    isNone = false;
  }

  bool is_some() const {
    return not isNone;
  }

  bool is_none() const {
    return isNone;
  }

  explicit operator const T&(){
    assert(isNone == false);
    return v.t;
  }
};

vector<cl::Device> getDevices(){
  vector<cl::Device> ret;

  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()){
    cout << "Failed to get platforms" << endl;
    return ret;
  }

  for (auto p = platforms.begin(); p != platforms.end(); ++p){
    vector<cl::Device> pdevs;
    try {
      p->getDevices(CL_DEVICE_TYPE_GPU, &pdevs);
      for (auto d = pdevs.begin(); d != pdevs.end(); ++d){
        if (not d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

        string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();
        if (ext.find("cl_khr_fp64") == string::npos && ext.find("cl_amd_fp64") == string::npos)
          continue;
        ret.push_back(*d);
      }
    } catch (...){
    }
  }

  return ret;
}

int main(){
  Mtx a(SZ, SZ), b(SZ, SZ), c(SZ, SZ), d(SZ, SZ);

  generate_sample(a, b, c);

  vector<cl::Device> devs = getDevices();

  cl::Context context(devs);
  cl::CommandQueue queue(context, devs[0]);
//  cl::Event event;
//  cl_int err;

  /* Setup clblas. */
//  err = clblasSetup();
//  if (err != CL_SUCCESS) {
//      printf("clblasSetup() failed with %d\n", err);
//      return 1;
//  }

  clock_t timing_start = clock();

  cl::Buffer da(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.rows * a.cols * sizeof(double), a.data);
  cl::Buffer db(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.rows * b.cols * sizeof(double), b.data);
  cl::Buffer dc(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c.rows * c.cols * sizeof(double), c.data);

//  cl_int code = clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
//                            a.rows, b.cols, a.cols,
//                            1., da(), 0, a.cols,
//                            db(), 0, b.cols,
//                            1., dc(), 0, c.cols,
//                            1, &queue(),
//                            0, NULL,
//                            &event());

   StatusCode code = Gemm(Layout::kRowMajor, Transpose::kNo, Transpose::kNo,
                          a.rows, b.cols, a.cols,
                          1., da(), 0, a.cols,
                          db(), 0, b.cols,
                          1., dc(), 0, c.cols,
                          &queue(), NULL);

//  if (code != CL_SUCCESS){
  if (code != StatusCode::kSuccess){
    cout << "clblasDgemm failed with " << (int)code << endl;
  } else {
//    err = event.wait();
    queue.enqueueReadBuffer(dc, CL_TRUE, 0, c.rows * c.cols * sizeof(double), c.data);
  }

  cout << "Time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

//  clblasTeardown();

//  for (size_t i = 0; i < SZ; ++i){
//    for (size_t j = 0; j < SZ; ++j)
//      cout << c.data[i * SZ + j] << " ";
//    cout << endl;
//  }

  return 0;
}

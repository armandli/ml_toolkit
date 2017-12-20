#include <cstdlib>
#include <ctime>
#include <vector>
#include <future>
#include <algorithm>
#include <iostream>

#include <x86intrin.h>

using namespace std;

#define PERF_SIZE 1000

using Dstp = double*;
using Srcp = const double*;

//allows memory to be not 32 byte aligned
void single_thread_add1(Dstp dst, Srcp s1, Srcp s2, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir)
    for (size_t ic = 0; ic < cols; ic += 4){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_add_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

void test1(){
  Dstp a = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp b = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp r = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);

  for (size_t i = 0; i < 2048 * 2048; ++i){
    a[i] = i;
    b[i] = i + 10;
  }

  clock_t sum = 0;
  for (size_t i = 0; i < PERF_SIZE; ++i){
    clock_t start = clock();
    single_thread_add1(r, a, b, 2048, 2048, 2048);
    sum += clock() - start;
  }

  cout << "STA: " << sum / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

//disaalow memory to be 32 byte aligned, and avoid write cache
//this is twice as fast as add1
void single_thread_add2(Dstp dst, Srcp s1, Srcp s2, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir)
    for (size_t ic = 0; ic < cols; ic += 4){
      __m256d a = _mm256_load_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_load_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_add_pd(a, b);
      _mm256_stream_pd(&dst[ir * colstride + ic], r);
    }
}

void test2(){
  Dstp a = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp b = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp r = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);

  for (size_t i = 0; i < 2048 * 2048; ++i){
    a[i] = i;
    b[i] = i + 10;
  }

  clock_t sum = 0;
  for (size_t i = 0; i < PERF_SIZE; ++i){
    clock_t start = clock();
    single_thread_add2(r, a, b, 2048, 2048, 2048);
    sum += clock() - start;
  }

  cout << "STA2: " << sum / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

#define TSIZE 8
void multi_thread_add1(Dstp dst, Srcp s1, Srcp s2, size_t rows, size_t colstride){
  future<void> futures[TSIZE];

  for (size_t i = 0; i < TSIZE; ++i){
    futures[i] = async(launch::async, [i, dst, s1, s2, rows, colstride](){
        size_t blocksize = rows * colstride / TSIZE;
        blocksize += (rows * colstride) & (TSIZE - 1) ? 1 : 0;
        size_t block_start = i * blocksize;
        size_t block_end = std::min((i + 1) * blocksize, rows * colstride);
        for (size_t i = block_start; i < block_end; i += 4){
          __m256d a = _mm256_load_pd(&s1[i]);
          __m256d b = _mm256_load_pd(&s2[i]);
          __m256d r = _mm256_add_pd(a, b);
          _mm256_stream_pd(&dst[i], r);
        }
    });
  }
  for (auto& f : futures) f.get();
}

void test3(){
  Dstp a = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp b = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);
  Dstp r = (double*)aligned_alloc(32, sizeof(double) * 2048 * 2048);

  for (size_t i = 0; i < 2048 * 2048; ++i){
    a[i] = i;
    b[i] = i + 10;
  }

  clock_t sum = 0;
  for (size_t i = 0; i < PERF_SIZE; ++i){
    clock_t start = clock();
    multi_thread_add1(r, a, b, 2048, 2048);
    sum += clock() - start;
  }

  cout << "MTA: " << sum / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}


int main(){
  int num_cpus = std::thread::hardware_concurrency();
  cout << num_cpus << endl;

  test1();
  test2();
  test3();
}

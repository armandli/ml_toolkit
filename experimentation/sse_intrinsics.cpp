#include <x86intrin.h>

#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

void test_compare_nan(){
  __m256d nanv = _mm256_set1_pd(nan(""));
  __m256d onev = _mm256_set1_pd(1.);
  double array[4] = {1., 2., nan(""), 3.};
  __m256d v = _mm256_loadu_pd(array);

  __m256d t1 = _mm256_cmp_pd(onev, nanv, _CMP_EQ_OQ);
  __m256d t2 = _mm256_cmp_pd(v, nanv, _CMP_EQ_OQ);
  __m256d t3 = _mm256_cmp_pd(nanv, nanv, _CMP_EQ_OQ);
  __m256d t4 = _mm256_cmp_pd(v, v, _CMP_EQ_OQ);

  double t1a[4];
  double t2a[4];
  double t3a[4];
  double t4a[4];
  _mm256_storeu_pd(t1a, t1);
  _mm256_storeu_pd(t2a, t2);
  _mm256_storeu_pd(t3a, t3);
  _mm256_storeu_pd(t4a, t4);

  for (size_t i = 0; i < 4; ++i)
    cout << t1a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t2a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t3a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t4a[i] << " ";
  cout << endl;
}

void test_div_nan(){
  __m256d nanv = _mm256_set1_pd(nan(""));
  double array1[4] = {1., 2., nan(""), 3.};
  __m256d v1 = _mm256_loadu_pd(array1);
  double array2[4] = {1., 0., nan(""), 3.};
  __m256d v2 = _mm256_loadu_pd(array2);

  __m256d t1 = _mm256_div_pd(v1, nanv);
  __m256d t2 = _mm256_div_pd(v2, nanv);
  __m256d t3 = _mm256_div_pd(v1, v2);
  __m256d t4 = _mm256_div_pd(v2, v1);

  double t1a[4];
  double t2a[4];
  double t3a[4];
  double t4a[4];
  _mm256_storeu_pd(t1a, t1);
  _mm256_storeu_pd(t2a, t2);
  _mm256_storeu_pd(t3a, t3);
  _mm256_storeu_pd(t4a, t4);

  for (size_t i = 0; i < 4; ++i)
    cout << t1a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t2a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t3a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << t4a[i] << " ";
  cout << endl;

}

__m256d implement_mm256_isnan_pd(__m256d x){
  long long spattern = 0x7FF0000000000000LL;
  long long sfilter  = 0x000FFFFFFFFFFFFFLL;
  __m256i snan = _mm256_set1_epi64x(spattern);
  __m256i sfil = _mm256_set1_epi64x(sfilter);
  __m256i zero = _mm256_set1_epi64x(0);

  __m256i cx = _mm256_castpd_si256(x);
  __m256i sx = _mm256_and_si256(cx, snan);
  __m256i fx = _mm256_and_si256(cx, sfil);

  __m256i c1 = _mm256_cmpeq_epi64(sx, snan);
  __m256i c2 = _mm256_cmpgt_epi64(fx, zero);
  __m256i r1  = _mm256_and_si256(c1, c2);

  __m256d r = _mm256_castsi256_pd(r1);
  return r;
}

void test_mm256_isnan_pd(){
  double t1a[4] = {numeric_limits<double>::infinity(), numeric_limits<double>::infinity() * -1., numeric_limits<double>::quiet_NaN(), numeric_limits<double>::signaling_NaN()};
  double t2a[4] = {1., -1., 157., -157.};
  __m256d t1 = _mm256_loadu_pd(t1a);
  __m256d t2 = _mm256_loadu_pd(t2a);

  __m256d r1 = implement_mm256_isnan_pd(t1);
  __m256d r2 = implement_mm256_isnan_pd(t2);

  double r1a[4], r2a[4];
  _mm256_storeu_pd(r1a, r1);
  _mm256_storeu_pd(r2a, r2);

  for (size_t i = 0; i < 4; ++i)
    cout << r1a[i] << " ";
  cout << endl;

  for (size_t i = 0; i < 4; ++i)
    cout << r2a[i] << " ";
  cout << endl;
}

//implement using taylor series e^x = 1 + x + x^2 / 2! + x^3 / 3! + x^4 / 4! ...
//TODO: this is buggy for large negative values
__m256d vanilla_mm256_exp_pd(__m256d x){
  __m256d p1 = _mm256_set1_pd(1.);
  __m256d precision = _mm256_set1_pd(1.11e-16);

  __m256d n = _mm256_set1_pd(0.);
  __m256d x1 = _mm256_set1_pd(1.);
  __m256d sum = _mm256_set1_pd(0.);
  unsigned long long test = 0;
  int count = 0;
  do {
    sum = _mm256_add_pd(sum, x1);
    n = _mm256_add_pd(n, p1);
    __m256d a = _mm256_div_pd(x, n);
    x1 = _mm256_mul_pd(x1, a);
    __m256d t = _mm256_cmp_pd(x1, precision, _CMP_GT_OQ);
    unsigned long long ta[4];
    _mm256_storeu_pd((double*)ta, t);
    test = 0;
    for (size_t i = 0; i < 4; ++i)
      test |= ta[i];
    count++;
  } while (test != 0ULL);

  cout << "vanilla mm256_exp_pd total loop count: " << count << endl;

  return sum;
}

void test_vanilla_mm256_exp_pd(){
  double tresa[4];

  double t1a[4] = {0., 1., 3.6, 1.746};
  __m256d t1v = _mm256_loadu_pd(t1a);

  __m256d tres = vanilla_mm256_exp_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double  texp1[4] = {exp(0.), exp(1.), exp(3.6), exp(1.746)};

  for (size_t i = 0; i < 4; ++i){
    cout << tresa[i] << " " << texp1[i] << endl;
  }

  double t2a[4] = {0.5, -0.5, 15., -15.};
  t1v = _mm256_loadu_pd(t2a);

  tres = vanilla_mm256_exp_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double texp2[4] = {exp(0.5), exp(-0.5), exp(15.), exp(-15.)};
  for (size_t i = 0; i < 4; ++i)
    cout << tresa[i] << " " << texp2[i] << endl;

  double t3a[4] = {28.76, -28.76, 97.7778, -97.7778};
  t1v = _mm256_loadu_pd(t3a);

  tres = vanilla_mm256_exp_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double texp3[4] = {exp(28.76), exp(-28.76), exp(97.7778), exp(-97.7778)};
  for (size_t i = 0; i < 4; ++i)
    cout << tresa[i] << " " << texp3[i] << endl;
}

//e^x = limit n -> inf (1 + x / n) ^ n
__m256d fast_mm256_exp256_pd(__m256d x){
  __m256d v256 = _mm256_set1_pd(256.);
  __m256d v1   = _mm256_set1_pd(1.);

  __m256d k = _mm256_div_pd(x, v256);
  k = _mm256_add_pd(k, v1);

  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);

  return k;
}

void test_fast_mm256_exp256_pd(){
  double t1a[4] = {0., 1., 3.6, 1.746};
  __m256d t1v = _mm256_loadu_pd(t1a);

  __m256d tres = fast_mm256_exp256_pd(t1v);
  double tresa[4];
  _mm256_storeu_pd(tresa, tres);

  double  texp[4] = {exp(0.), exp(1.), exp(3.6), exp(1.746)};

  for (size_t i = 0; i < 4; ++i){
    cout << tresa[i] << " " << texp[i] << endl;
  }
}

__m256d fast_mm256_exp1024_pd(__m256d x){
  __m256d v1024 = _mm256_set1_pd(1024.);
  __m256d v1    = _mm256_set1_pd(1.);

  __m256d k = _mm256_div_pd(x, v1024);
  k = _mm256_add_pd(k, v1);

  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);
  k = _mm256_mul_pd(k, k);

  return k;
}

void test_fast_mm256_exp1024_pd(){
  double tresa[4];

  double t1a[4] = {0., 1., 3.6, 1.746};
  __m256d t1v = _mm256_loadu_pd(t1a);

  __m256d tres = fast_mm256_exp1024_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double  texp1[4] = {exp(0.), exp(1.), exp(3.6), exp(1.746)};

  for (size_t i = 0; i < 4; ++i){
    cout << tresa[i] << " " << texp1[i] << endl;
  }

  double t2a[4] = {0.5, -0.5, 15., -15.};
  t1v = _mm256_loadu_pd(t2a);

  tres = fast_mm256_exp1024_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double texp2[4] = {exp(0.5), exp(-0.5), exp(15.), exp(-15.)};
  for (size_t i = 0; i < 4; ++i)
    cout << tresa[i] << " " << texp2[i] << endl;

  double t3a[4] = {28.76, -28.76, 97.7778, -97.7778};
  t1v = _mm256_loadu_pd(t3a);

  tres = fast_mm256_exp1024_pd(t1v);
  _mm256_storeu_pd(tresa, tres);

  double texp3[4] = {exp(28.76), exp(-28.76), exp(97.7778), exp(-97.7778)};
  for (size_t i = 0; i < 4; ++i)
    cout << tresa[i] << " " << texp3[i] << endl;
}


// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
__m256d implement_mm256_tanh_pd(__m256d x){
  __m256d v1 = _mm256_set1_pd(1.);
  __m256d v2 = _mm256_set1_pd(2.);
  __m256d x2 = _mm256_mul_pd(x, v2);
  __m256d exp2x = fast_mm256_exp1024_pd(x2);
  __m256d nom = _mm256_sub_pd(exp2x, v1);
  __m256d dnm = _mm256_add_pd(exp2x, v1);
  __m256d r = _mm256_div_pd(nom, dnm);
  return r;
}

void test_mm256_tanh_pd(){
  double t1a[4] = {0., 1., 3.6, 1.746};
  __m256d t1v = _mm256_loadu_pd(t1a);

  __m256d tres = implement_mm256_tanh_pd(t1v);
  double tresa[4];
  _mm256_storeu_pd(tresa, tres);

  double  texp[4] = {tanh(0.), tanh(1.), tanh(3.6), tanh(1.746)};

  for (size_t i = 0; i < 4; ++i){
    cout << tresa[i] << " " << texp[i] << endl;
  }
}

void test_fma(){
  __m256d v1 = _mm256_set1_pd(7.);
  __m256d v2 = _mm256_set1_pd(10.);
  __m256d v3 = _mm256_set1_pd(2.);

  __m256d t1 = _mm256_fmadd_pd(v1, v2, v3);

  double t1a[4];
  _mm256_storeu_pd(t1a, t1);

  for (size_t i = 0; i < 4; ++i)
    cout << t1a[i] << " ";
  cout << endl;
}

int main(){
  cout << "test_compare_nan" << endl;
  test_compare_nan();
  cout << "test_div_nan" << endl;
  test_div_nan();
  cout << "test_fma" << endl;
  test_fma();
  cout << "test_vanilla_mm256_exp_pd" << endl;
  test_vanilla_mm256_exp_pd();
  cout << "test_fast_mm256_exp256_pd" << endl;
  test_fast_mm256_exp256_pd();
  cout << "test_fast_mm256_exp1024_pd" << endl;
  test_fast_mm256_exp1024_pd();
  cout << "test mm256_tanh_pd" << endl;
  test_mm256_tanh_pd();
  cout << "test mm256_isnan_pd" << endl;
  test_mm256_isnan_pd();
}

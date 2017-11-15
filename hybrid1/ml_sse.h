#ifndef ML_SSE
#define ML_SSE

#include <x86intrin.h>

#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

#include <ml_common.h>

//TODO: load and store can assume 32bit alignment here
//TODO: there may be missing operations
//TODO: rename each function to conform with cuda implementations
//TODO: due to submatrix operation, we need a equivalent 2D operation for each 1D operation

namespace ML::SSE {

namespace {
const long long oned  = 0xFFFFFFFFFFFFFFFFLL;
const long long poned = 0x7FFFFFFFFFFFFFFFLL;

const __m256d pd1 = _mm256_set1_pd(1.);
const __m256d nd1 = _mm256_set1_pd(-1.);
const __m256d zd1 = _mm256_set1_pd(0.);
} // namespace

namespace SPPL {

__m256d _mm256_isnan_pd(__m256d x){
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

__m256d _mm256_fexp_pd(__m256d x){
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

__m256d _mm256_exp_pd(__m256d x){
  __m256d v1048576 = _mm256_set1_pd(1048576.);
  __m256d v1    = _mm256_set1_pd(1.);

  __m256d k = _mm256_div_pd(x, v1048576);
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

__m256d _mm256_tanh_pd(__m256d x){
  __m256d v1 = _mm256_set1_pd(1.);
  __m256d v2 = _mm256_set1_pd(2.);
  __m256d x2 = _mm256_mul_pd(x, v2);
  __m256d exp2x = _mm256_fexp_pd(x2); //NOTE: using _mm256_exp_pd could drastically increase precision
  __m256d nom = _mm256_sub_pd(exp2x, v1);
  __m256d dnm = _mm256_add_pd(exp2x, v1);
  __m256d r = _mm256_div_pd(nom, dnm);
  return r;
}
} // SPPL

using Dstp = double* const;
using Srcp = const double* const;

void const_init_2d_sse_pd(Dstp dst, double v, size_t rows, size_t cols, size_t rowstride, size_t colstride){
  double col_boundary[MTX_BLOCK_RSZ] = {0.};
  for (size_t i = 0; i < (cols & MTX_BLOCK_RMASK); ++i)
    col_boundary[i] = v;
  __m256d b256 = _mm256_setr_pd(col_boundary[0], col_boundary[1], col_boundary[2], col_boundary[3]);
  __m256d v256 = _mm256_set1_pd(v);

  if (cols < colstride){
    for (size_t ir = 0; ir < rows; ++ir){
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ)
        _mm256_storeu_pd(&dst[ir * colstride + ic], v256);
      _mm256_storeu_pd(&dst[ir * colstride + (cols & ~MTX_BLOCK_RMASK)], b256);
    }
  } else {
    for (size_t ir = 0; ir < rows; ++ir)
      for(size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ)
        _mm256_storeu_pd(&dst[ir * colstride + ic], v256);
  }

  v256 = _mm256_setzero_pd();
  for (size_t ir = rows; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ)
      _mm256_storeu_pd(&dst[ir * colstride + ic], v256);
}

void transpose4x4_2d_sse_pd(double* __restrict__ const dst, const double* __restrict__ const src, size_t rowstride, size_t colstride){
  for (size_t i = 0; i < rowstride; i += MTX_BLOCK_RSZ)
    for (size_t j = 0; j < colstride; j += MTX_BLOCK_RSZ){
      double* d = &dst[j * rowstride + i];
      const double* s = &src[i * colstride + j];

      __m256d r1 = _mm256_loadu_pd(&s[colstride * 0]);
      __m256d r2 = _mm256_loadu_pd(&s[colstride * 1]);
      __m256d r3 = _mm256_loadu_pd(&s[colstride * 2]);
      __m256d r4 = _mm256_loadu_pd(&s[colstride * 3]);
    
      __m256d t1 = _mm256_unpacklo_pd(r1, r2);
      __m256d t2 = _mm256_unpackhi_pd(r1, r2);
      __m256d t3 = _mm256_unpacklo_pd(r3, r4);
      __m256d t4 = _mm256_unpackhi_pd(r3, r4);
    
      __m256d p1 = _mm256_permute2f128_pd(t1, t3, 0x20);
      __m256d p2 = _mm256_permute2f128_pd(t2, t4, 0x20);
      __m256d p3 = _mm256_permute2f128_pd(t1, t3, 0x31);
      __m256d p4 = _mm256_permute2f128_pd(t2, t4, 0x31);
    
      _mm256_storeu_pd(&d[rowstride * 0], p1);
      _mm256_storeu_pd(&d[rowstride * 1], p2);
      _mm256_storeu_pd(&d[rowstride * 2], p3);
      _mm256_storeu_pd(&d[rowstride * 3], p4);
    }
}

void add_1d_sse_pd(Dstp dst, Srcp s1, Srcp s2, size_t rowstride, size_t colstride){
  for (size_t ir = 0; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_add_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

void add_const_2d_sse_pd(Dstp dst, Srcp src, double v, size_t rows, size_t cols, size_t colstride){
  double c[MTX_BLOCK_RSZ] = {0.};
  for (size_t i = 0; i < (cols & MTX_BLOCK_RMASK); ++i)
    c[i] = v;
  __m256d b256 = _mm256_setr_pd(c[0], c[1], c[2], c[3]);
  __m256d v256 = _mm256_set1_pd(v);

  if (cols < colstride){
    for (size_t ir = 0; ir < rows; ++ir){
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_add_pd(s, v256);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + (cols & ~MTX_BLOCK_RMASK)]);
      __m256d r = _mm256_add_pd(s, b256);
      _mm256_storeu_pd(&dst[ir * colstride + (cols & ~MTX_BLOCK_RMASK)], r);
    }
  } else {
    for (size_t ir = 0; ir < rows; ++ir)
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_add_pd(s, v256);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
  }
}

void sub_1d_sse_pd(Dstp dst, Srcp s1, Srcp s2, size_t rowstride, size_t colstride){
  for (size_t ir = 0; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_sub_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

void sub_mc_2d_sse_pd(Dstp dst, Srcp src, double v, size_t rows, size_t cols, size_t colstride){
  double c[MTX_BLOCK_RSZ] = {0.};
  for (size_t i = 0; i < (cols & MTX_BLOCK_RMASK); ++i)
    c[i] = v;
  __m256d b256 = _mm256_setr_pd(c[0], c[1], c[2], c[3]);
  __m256d v256 = _mm256_set1_pd(v);

  if (cols < colstride){
    for (size_t ir = 0; ir < rows; ++ir){
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_sub_pd(s, v256);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + (cols & ~MTX_BLOCK_RMASK)]);
      __m256d r = _mm256_sub_pd(s, b256);
      _mm256_storeu_pd(&dst[ir * colstride + (cols & ~MTX_BLOCK_RMASK)], r);
    }
  } else {
    for (size_t ir = 0; ir < rows; ++ir)
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_sub_pd(s, v256);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
  }
}

void sub_cm_2d_sse_pd(double v, Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  double c[MTX_BLOCK_RSZ] = {0.};
  for (size_t i = 0; i < (cols & MTX_BLOCK_RMASK); ++i)
    c[i] = v;
  __m256d b256 = _mm256_setr_pd(c[0], c[1], c[2], c[3]);
  __m256d v256 = _mm256_set1_pd(v);

  if (cols < colstride){
    for (size_t ir = 0; ir < rows; ++ir){
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_sub_pd(v256, s);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + (cols & ~MTX_BLOCK_RMASK)]);
      __m256d r = _mm256_sub_pd(b256, s);
      _mm256_storeu_pd(&dst[ir * colstride + (cols & ~MTX_BLOCK_RMASK)], r);
    }
  } else {
    for (size_t ir = 0; ir < rows; ++ir)
      for(size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
        __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
        __m256d r = _mm256_sub_pd(v256, s);
        _mm256_storeu_pd(&dst[ir * colstride + ic], r);
      }
  }
}

void emul_1d_sse_pd(Dstp dst, Srcp s1, Srcp s2, size_t rowstride, size_t colstride){
  for (size_t ir = 0; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_mul_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

void emul_const_2d_sse_pd(Dstp dst, Srcp src, double v, size_t rowstride, size_t colstride){
  __m256d v256 = _mm256_set1_pd(v);
  for (size_t ir = 0; ir < rowstride; ++ir)
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
      __m256d r = _mm256_mul_pd(s, v256);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
}

//TODO: may need to protect against s2 = 0. or nan value
void ediv_2d_sse_pd(Dstp dst, Srcp s1, Srcp s2, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    for (size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&s1[ir * colstride + ic]);
      __m256d b = _mm256_loadu_pd(&s2[ir * colstride + ic]);
      __m256d r = _mm256_div_pd(a, b);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
    for (size_t ic = (cols & ~MTX_BLOCK_RMASK); ic < cols; ++ic)
      dst[ir * colstride + ic] = s1[ir * colstride + ic] / s2[ir * colstride + ic];
  }
}

void ediv_mc_2d_sse_pd(Dstp dst, Srcp src, double v, size_t rows, size_t cols, size_t colstride){
  __m256d v256 = _mm256_set1_pd(v);
  for (size_t ir = 0; ir < rows; ++ir){
    for (size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
      __m256d r = _mm256_div_pd(s, v256);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
    for (size_t ic = (cols & ~MTX_BLOCK_RSZ); ic < cols; ++ic)
      dst[ir * colstride + ic] = src[ir * colstride + ic] / v;
  }
}

void ediv_cm_2d_sse_pd(double v, Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  __m256d v256 = _mm256_set1_pd(v);
  for (size_t ir = 0; ir < rows; ++ir){
    for (size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
      __m256d s = _mm256_loadu_pd(&src[ir * colstride + ic]);
      __m256d r = _mm256_div_pd(v256, s);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
    for (size_t ic = (cols & ~MTX_BLOCK_RSZ); ic < cols; ++ic)
      dst[ir * colstride + ic] = v / src[ir * colstride + ic];
  }
}

//TODO: do not expose this interface
double max_row_coeff_pd(Srcp src, size_t cols){
  __m256d maxv = _mm256_set1_pd(std::numeric_limits<double>::min());
  for (size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&src[ic]);
    maxv = _mm256_max_pd(maxv, a);
  }
  double maxa[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(maxa, maxv);
  double mv = *std::max_element(maxa, maxa + MTX_BLOCK_RSZ);
  for (size_t ic = (cols & ~MTX_BLOCK_RMASK); ic < cols; ++ic)
    mv = std::max(mv, src[ic]);
  return mv;
}

void max_row_coeffs_2d_sse_pd(Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    double mv = max_row_coeff_pd(&src[ir * colstride], cols);
    dst[ir] = mv;
  }
}

//TODO: do not expose this function
double min_row_coeff_pd(Srcp src, size_t cols){
  __m256d minv = _mm256_set1_pd(std::numeric_limits<double>::max());
  for (size_t ic = 0; ic < (cols & ~MTX_BLOCK_RMASK); ic += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&src[ic]);
    minv = _mm256_min_pd(minv, a);
  }
  double mina[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(mina, minv);
  double mv = *std::min_element(mina, mina + MTX_BLOCK_RSZ);
  for (size_t ic = (cols & ~MTX_BLOCK_RMASK); ic < cols; ++ic)
    mv = std::min(mv, src[ic]);
  return mv;
}

void min_row_coeffs_2d_sse_pd(Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    double mv = min_row_coeff_pd(&src[ir * colstride], cols);
    dst[ir] = mv;
  }
}

double sum_row_pd(Srcp src, size_t colstride){
  __m256d sumv = zd1;
  for (size_t i = 0; i < colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&src[i]);
    sumv      = _mm256_add_pd(sumv, a);
  }
  double suma[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(suma, sumv);
  return std::accumulate(suma, suma + MTX_BLOCK_RSZ, 0.);
}

void sum_rows_2d_sse_pd(Dstp dst, Srcp src, size_t rows, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir)
    dst[ir] = sum_row_pd(&src[ir * colstride], colstride);
}

double sum_all_1d_sse_pd(Srcp src, size_t rows, size_t colstride){
  __m256d sumv = zd1;
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&src[i]);
    sumv      = _mm256_add_pd(sumv, a);
  }
  double suma[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(suma, sumv);
  return std::accumulate(suma, suma + MTX_BLOCK_RSZ, 0.);
}

void mean_rows_1d_sse_pd(Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    double s = sum_row_pd(&src[ir * colstride], colstride);
    dst[ir] = s / (double)cols;
  }
}

double mean_all_1d_sse_pd(Srcp src, size_t rows, size_t cols, size_t colstride){
  double s = sum_all_1d_sse_pd(src, rows, colstride);
  return s / ((double)rows * (double)cols);
}

void sigmoid_2d_sse_pd(Dstp dst, Srcp m, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    for (size_t ic = 0; ic < cols; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&m[ir * colstride + ic]);
      __m256d b = _mm256_mul_pd(a, nd1);
      __m256d c = SPPL::_mm256_fexp_pd(b); //NOTE: using _mm256_exp_pd could drastically increase precision
      __m256d d = _mm256_add_pd(pd1, c);
      __m256d r = _mm256_div_pd(pd1, d);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
    for (size_t ic = (cols & ~MTX_BLOCK_RMASK); ic < cols; ++ic){
      double a = m[ir * colstride + ic];
      dst[ir * colstride + ic] = 1. / (1. + exp(a * -1.));
    }
  }
}

void tanh_1d_sse_pd(Dstp dst, Srcp m, size_t rows, size_t colstride){
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&m[i]);
    __m256d r = SPPL::_mm256_tanh_pd(a);
    _mm256_storeu_pd(&dst[i], r);
  }
}

void relu_1d_sse_pd(Dstp dst, Srcp m, size_t rows, size_t colstride){
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&m[i]);
    __m256d b = _mm256_cmp_pd(a, zd1, _CMP_GT_OQ);
    __m256d r = _mm256_and_pd(a, b);
    _mm256_storeu_pd(&dst[i], r);
  }
}

void drelu_1d_sse_pd(Dstp dst, Srcp dm, Srcp m, size_t rows, size_t colstride){
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&dm[i]);
    __m256d b = _mm256_loadu_pd(&m[i]);
    __m256d c = _mm256_cmp_pd(b, zd1, _CMP_GT_OQ);
    __m256d r = _mm256_and_pd(a, c);
    _mm256_storeu_pd(&dst[i], r);
  }
}

double loss_l2_1d_sse_pd(Srcp m, double reg, size_t rows, size_t colstride){
  __m256d sum = zd1;
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&m[i]);
    __m256d r = _mm256_mul_pd(a, a);
    sum = _mm256_add_pd(sum, r);
  }

  double r[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(r, sum);
  double loss = std::accumulate(r, r + MTX_BLOCK_RSZ, 0.);
  return loss * 0.5 * reg;
}

//NOTE: softmax can only handle matrix value within 0.0 and 1.0, and 
//result is only has 2 sig fig
void softmax_r_2d_sse_pd(Dstp dst, Srcp src, size_t rows, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir){
    __m256d rowsum = zd1;
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&src[ir * colstride + ic]);
      a = SPPL::_mm256_fexp_pd(a);
      rowsum = _mm256_add_pd(rowsum, a);
    }

    double rowsuma[MTX_BLOCK_RSZ];
    _mm256_storeu_pd(rowsuma, rowsum);
    double sum = std::accumulate(rowsuma, rowsuma + MTX_BLOCK_RSZ, 0.);

    __m256d sumv = _mm256_set1_pd(sum);
    for (size_t ic = 0; ic < colstride; ic += MTX_BLOCK_RSZ){
      __m256d a = _mm256_loadu_pd(&src[ir * colstride + ic]);
      a = SPPL::_mm256_fexp_pd(a);
      __m256d r = _mm256_div_pd(a, sumv);
      _mm256_storeu_pd(&dst[ir * colstride + ic], r);
    }
  }
}

//TODO: do not expose this function
double diff_square_sum_pd(Srcp o, Srcp y, size_t rows, size_t colstride){
  __m256d sum = _mm256_set1_pd(0.);
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&o[i]);
    __m256d b = _mm256_loadu_pd(&y[i]);
    __m256d c = _mm256_sub_pd(a, b);
    __m256d r = _mm256_mul_pd(c, c);
    sum       = _mm256_add_pd(sum, r);
  }

  double suma[MTX_BLOCK_RSZ];
  _mm256_storeu_pd(suma, sum);
  return std::accumulate(suma, suma + MTX_BLOCK_RSZ, 0.);
}

double mse_loss_1d_sse_pd(Srcp o, Srcp y, size_t rows, size_t colstride){
  return diff_square_sum_pd(o, y, rows, colstride) / (double)rows;
}

double mse_accuracy_1d_sse_pd(Srcp o, Srcp y, size_t rows, size_t colstride){
  return sqrt(diff_square_sum_pd(o, y, rows, colstride)) * 0.5 / (double)rows;
}

void deriviative_row_1d_sse_pd(Dstp dst, Srcp o, Srcp y, size_t rows, size_t colstride){
  __m256i oi = _mm256_set1_epi64x(oned);
  __m256d ov = _mm256_castsi256_pd(oi);
  __m256d sz = _mm256_set1_pd((double)rows);
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&o[i]);
    __m256d b = _mm256_loadu_pd(&y[i]);
    __m256d c = _mm256_sub_pd(a, b);
    __m256d d = _mm256_div_pd(c, sz);
    __m256d t = SPPL::_mm256_isnan_pd(d);
    __m256d t2= _mm256_xor_pd(t, ov);
    __m256d r = _mm256_and_pd(d, t2);
    _mm256_storeu_pd(&dst[i], r);
  }
}

//TODO: perhaps comparison with tolerance here is still not completely correct
bool block_cmp_equal_1d_sse_pd(Srcp x, Srcp y, double epsilon, size_t rows, size_t colstride){
  __m256i mi = _mm256_set1_epi64x(poned);
  __m256i oi = _mm256_set1_epi64x(oned);
  __m256d cr = _mm256_castsi256_pd(oi);
  __m256d mask = _mm256_castsi256_pd(mi);
  __m256d thre = _mm256_set1_pd(epsilon);
  for (size_t i = 0; i < rows * colstride; i += MTX_BLOCK_RSZ){
    __m256d a = _mm256_loadu_pd(&x[i]);
    __m256d b = _mm256_loadu_pd(&y[i]);
    __m256d d = _mm256_sub_pd(a, b);
    __m256d m = _mm256_and_pd(d, mask);
    __m256d t = _mm256_cmp_pd(m, thre, _CMP_LE_OQ);
    cr = _mm256_and_pd(cr, t);
  }
  __m256i res = _mm256_castpd_si256(cr);
  long long resa[MTX_BLOCK_RSZ];
  _mm256_storeu_si256((__m256i*)resa, res);
  for (size_t i = 0; i < MTX_BLOCK_RSZ; ++i)
    if (resa[i] != oned)
      return false;
  return true;
}

} //ML::SSE

#endif//ML_SSE

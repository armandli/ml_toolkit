#ifndef ML_MTXOP
#define ML_MTXOP

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>

#include <ml_common.h>

namespace ML::MTXOP {

using Dstp = double* const;
using Srcp  = const double* const;

void rnd_uniform_init_2d_mtxop_pd(Dstp dst, double lb, double ub, size_t rows, size_t cols, size_t colstride){
  srand(time(NULL));
  for (size_t ir = 0; ir < rows; ++ir){
    std::for_each(&dst[ir * colstride], &dst[ir * colstride + cols], [lb, ub](double& d){
        d = (double)rand() / RAND_MAX * (ub - lb) + lb;
    });
  }
}

void rnd_normal_init_2d_mtxop_pd(Dstp dst, double ex, double sd, size_t rows, size_t cols, size_t colstride){
  std::default_random_engine& eng = ML::get_default_random_engine();
  std::normal_distribution<double> dist(ex, sd);

  for (size_t ir = 0; ir < rows; ++ir)
    std::for_each(&dst[ir * colstride], &dst[ir * colstride + cols], [&eng, &dist](double& d){
        d = dist(eng);
    });
}

//TODO: do not expose this function
size_t max_row_coeff_idx_2d_mtxop_pd(Srcp m, size_t row, size_t cols, size_t colstride){
  Srcp elem = std::max_element(&m[row * colstride], &m[row * colstride + cols]);
  assert(elem != nullptr);
  return (elem - &m[row * colstride]) / sizeof(double);
}

void msvm_loss_2d_mtxop_pd(Dstp dst, Srcp o, Srcp y, double f, size_t rows, size_t cols, size_t colstride){
  double loss = 0.;
  for (size_t ir = 0; ir < rows; ++ir){
    size_t midx = max_row_coeff_idx_2d_mtxop_pd(y, ir, cols, colstride);
    double v = o[ir * colstride + midx];
    double sum = 0.;
    std::for_each(&o[ir * colstride], &o[ir * colstride + cols], [&sum, &v, &f](double d){
        double e = d - v + f;
        if (e < 0.) e = 0.;
        sum += e * e;
    });
    loss += sum - f * f;
  }
  *dst = loss / (double)rows;
}

void ce_loss_2d_mtxop_pd(Dstp dst, Srcp o, Srcp y, size_t rows, size_t cols, size_t colstride){
  double loss = 0.;
  for (size_t ir = 0; ir < rows; ++ir){
    size_t yidx = max_row_coeff_idx_2d_mtxop_pd(y, ir, cols, colstride);
    loss += -1. * std::log(o[ir * colstride + yidx]);
  }
  *dst = loss / (double)rows;
}

void ce_accuracy_2d_mtxop_pd(Dstp dst, Srcp o, Srcp y, size_t rows, size_t cols, size_t colstride){
  size_t count = 0;
  for (size_t ir = 0; ir < rows; ++ir){
    size_t oidx = max_row_coeff_idx_2d_mtxop_pd(o, ir, cols, colstride);
    size_t yidx = max_row_coeff_idx_2d_mtxop_pd(y, ir, cols, colstride);
    if (yidx == oidx)
      count++;
  }
  *dst = (double)count / (double)rows;
}

void add_cc_1d_mtxop_pd(Dstp dst, double s1, double s2){
  *dst = s1 + s2;
}

void sub_cc_1d_mtxop_pd(Dstp dst, double s1, double s2){
  *dst = s1 - s2;
}

void emul_cc_1d_mtxop_pd(Dstp dst, double s1, double s2){
  *dst = s1 * s2;
}

void ediv_cc_1d_mtxop_pd(Dstp dst, double s1, double s2){
  *dst = s1 / s2;
}

void sqrt_c_1d_mtxop_pd(Dstp dst, double src){
  *dst = std::sqrt(src);
}

void abs_2d_mtxop_pd(Dstp dst, Srcp src, size_t rows, size_t cols, size_t colstride){
  for (size_t ir = 0; ir < rows; ++ir)
    for (size_t ic = 0; ic < cols; ++ic)
      dst[ir * colstride + ic] = std::abs(src[ir * colstride + ic]);
}

} // ML::MTXOP

#endif//ML_MTXOP

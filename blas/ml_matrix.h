#ifndef ML_MATRIX
#define ML_MATRIX

//TODO: add cublas usages
//TODO: write delayed element-wise evaluation so to avoid unnecessary copying operation in long expressions
//TODO: cleanup interface

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <x86intrin.h>

#include <cblas.h>

#include <ml_common.h>

namespace ML {

enum MtxDim: int {
  MCol,
  MRow,
};

struct DimV {
  size_t idx;
  double val;
};

#define D1Iter(i, ib, ie) \
  for (size_t i = (ib); i < (ie); ++i)

#define D2IterR(ir, ic, rb, re, cb, ce) \
  for (size_t ir = (rb); ir < (re); ++ir) for (size_t ic = (cb); ic < (ce); ++ic) 

#define D2Idx(pt, ir, ic, rs, cs) \
  (pt)[(ir) * (cs) + (ic)]

class SubMtx {
  double* mData;
  size_t  mRows;
  size_t  mCols;
  size_t  mStride;

public:
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }

  SubMtx(double* data, size_t r, size_t c, size_t s):
    mData(data), mRows(r), mCols(c), mStride(s){}
  SubMtx(const SubMtx& o){
    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = D2Idx(o.mData, ir, ic, o.rows(), o.mStride);
  }
  SubMtx(SubMtx&& o){
    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = D2Idx(o.mData, ir, ic, o.rows(), o.mStride);
  }
  ~SubMtx() = default;

  SubMtx& operator=(const SubMtx& o){
    if (this == &o) return *this;

    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = D2Idx(o.mData, ir, ic, o.rows(), o.mStride);
    return *this;
  }
  SubMtx& operator=(SubMtx&& o){
    if (this == &o) return *this;

    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = D2Idx(o.mData, ir, ic, o.rows(), o.mStride);
    return *this;
  }

  double operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());

    return D2Idx(mData, i, j, 1, mStride);
  }
  double& operator()(size_t i, size_t j){
    assert(i < rows() && j < cols());

    return D2Idx(mData, i, j, 1, mStride);
  }
  
  template <typename F>
  SubMtx& unary_expr(F&& f){
    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = f(D2Idx(mData, ir, ic, rows(), mStride));
    return *this;
  }

  template <typename F>
  SubMtx& binary_expr(F&& f, const SubMtx& o){
    assert(rows() == o.rows() && cols() == o.cols());

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mStride) = f(D2Idx(mData, ir, ic, rows(), mStride),
                                                D2Idx(o.mData, ir, ic, o.rows(), o.mStride));
    return *this;
  }
};

/* SubMtx scalar assignment operations */
SubMtx& operator+=(SubMtx& a, double b){
  a.unary_expr([b](double v){ return v + b; });
  return a;
}
SubMtx& operator-=(SubMtx& a, double b){
  a.unary_expr([b](double v){ return v - b; });
  return a;
}

SubMtx& operator*=(SubMtx& a, double b){
  a.unary_expr([b](double v){ return v * b; });
  return a;
}

SubMtx& operator/=(SubMtx& a, double b){
  a.unary_expr([b](double v){
      if (b == 0.) return nan("");
      else         return v / b;
  });
  return a;
}

std::ostream& operator << (std::ostream& out, const SubMtx& m){
  for (size_t i = 0; i < m.rows(); ++i){
    bool is_first = true;
    for (size_t j = 0; j < m.cols(); ++j){
      if (is_first) is_first = false;
      else          out << " ";
      out << m(i, j);
    }
  }
  return out;
}

class Mtx {
  double* mData;
  size_t  mRows;
  size_t  mCols;
  size_t  mRowStride;
  size_t  mColStride;

  void zero_boundary(){
    D2IterR(ir, ic, 0, rows(), cols(), mColStride) D2Idx(mData, ir, ic, mRowStride, mColStride) = 0.;
    D2IterR(ir, ic, rows(), mRowStride, 0, mColStride) D2Idx(mData, ir, ic, mRowStride, mColStride) = 0.;
  }

  /* load matrix from fstream */
  void load(std::istream& in){
    assert(in.good());

    in.read((char*)&mRows, sizeof(size_t));
    in.read((char*)&mCols, sizeof(size_t));

    assert(mRows > 0 && mCols > 0);

    mRowStride = round_up(mRows);
    mColStride = round_up(mCols);

    mData = new double[mRowStride * mColStride];

    in.read((char*)mData, sizeof(double) * mRowStride * mColStride);
  }

  void transpose2x2_SSE(double *A, double *B, size_t lda, size_t ldb) const {
    __m128d r1 = _mm_load_pd(&A[0*lda]);
    __m128d r2 = _mm_load_pd(&A[1*lda]);
    __m128d t1 = _mm_unpacklo_pd(r1, r2);
    __m128d t2 = _mm_unpackhi_pd(r1, r2);
    _mm_store_pd(&B[0*ldb], t1);
    _mm_store_pd(&B[1*ldb], t2);
  }
  
  void transpose_block_SSE2x2(double *A, double *B, size_t rows, size_t cols, size_t lda, size_t ldb) const {
    for(size_t i = 0; i < rows; i += MTX_BLOCK_SZ) {
      for(size_t j = 0; j < cols; j += MTX_BLOCK_SZ) {
        size_t max_i2 = i + MTX_BLOCK_SZ;
        size_t max_j2 = j + MTX_BLOCK_SZ;
        for(size_t i2 = i; i2 < max_i2; i2 += MTX_MUL_DOUBLE_UNIT) {
          for(size_t j2 = j; j2 < max_j2; j2 += MTX_MUL_DOUBLE_UNIT) {
            transpose2x2_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
          }
        }
      }
    }
  }

public:
  Mtx(): mData(nullptr), mRows(0), mCols(0), mRowStride(0), mColStride(0) {}
  Mtx(size_t r, size_t c, double def = 0.):
    mData(nullptr), mRows(r), mCols(c), mRowStride(round_up(r)), mColStride(round_up(c)) {
    mData = new double[mRowStride * mColStride];
    D2IterR(ir, ic, 0, rows(), 0, cols()) D2Idx(mData, ir, ic, mRowStride, mColStride) = def;
    zero_boundary();
  }
  explicit Mtx(std::istream& in) {
    load(in);
  }
  explicit Mtx(const char* filename){
    std::ifstream in(filename, std::ifstream::in);
    assert(in.good());
    load(in);
    in.close();
  }
  Mtx(size_t r, size_t c, const std::vector<double>& data):
    mData(nullptr), mRows(r), mCols(c), mRowStride(round_up(r)), mColStride(round_up(c)) {
    assert(data.size() >= r * c);

    mData = new double[mRowStride * mColStride];

    for (size_t ir = 0, k = 0; ir < mRows; ++ir)
      for (size_t ic = 0; ic < mCols; ++ic, ++k)
        D2Idx(mData, ir, ic, mRowStride, mColStride) = data[k];
    zero_boundary();
  }
  Mtx(const Mtx& o):
    mData(new double[o.mRowStride * o.mColStride]), mRows(o.rows()), mCols(o.cols()),
    mRowStride(o.mRowStride), mColStride(o.mColStride){
    memcpy(mData, o.mData, sizeof(double) * o.mRowStride * o.mColStride);
  }
  Mtx(Mtx&& o):
    mData(o.mData), mRows(o.rows()), mCols(o.cols()), mRowStride(o.mRowStride), mColStride(o.mColStride) {
    o.mData = nullptr;
    o.mRows = o.mCols = o.mRowStride = o.mColStride = 0;
  }
  ~Mtx(){
    if (mData){
      delete[] mData;
      mData = nullptr;
    }
  }

  Mtx& operator=(const Mtx& o){
    if (this == &o) return *this;

    if (mData == nullptr && o.mData == nullptr){
      /* do nothing */
    } else if (mData == nullptr && o.mData != nullptr){
      mData = new double[o.mRowStride * o.mColStride];
      memcpy(mData, o.mData, o.mRowStride * o.mColStride * sizeof(double));
    } else if (o.mData == nullptr && mData != nullptr){
      delete[] mData;
      mData = nullptr;
    } else if (mRowStride * mColStride != o.mRowStride * o.mColStride){
      delete[] mData;
      mData = new double[o.mRowStride * o.mColStride];
      memcpy(mData, o.mData, o.mRowStride * o.mColStride * sizeof(double));
    } else {
      memcpy(mData, o.mData, o.mRowStride * o.mColStride * sizeof(double));
    }
    mRows = o.rows();
    mCols = o.cols();
    mRowStride = o.mRowStride;
    mColStride = o.mColStride;
    return *this;
  }
  Mtx& operator=(Mtx&& o){
    if (this == &o) return *this;

    if (mData != nullptr){
      delete[] mData;
      mData = nullptr;
    }
    if (o.mData != nullptr)
      mData = std::move(o.mData);

    mRows = std::move(o.rows());
    mCols = std::move(o.cols());
    mRowStride = std::move(o.mRowStride);
    mColStride = std::move(o.mColStride);
    o.mData = nullptr;
    o.mRows = o.mCols = o.mRowStride = o.mColStride = 0;
    return *this;
  }

  /* accessors */
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }
  double operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());

    return D2Idx(mData, i, j, mRowStride, mColStride);
  }
  double& operator()(size_t i, size_t j){
    assert(i < rows() && j < cols());

    return D2Idx(mData, i, j, mRowStride, mColStride);
  }
  SubMtx block(size_t rp, size_t cp, size_t rz = 1, size_t cz = 1) {
    assert(rp < rows() && cp < cols() && rp + rz <= rows() && cp + cz <= cols());

    double* data = &D2Idx(mData, rp, cp, mRowStride, mColStride);
    return SubMtx(data, rz, cz, mColStride);
  }
  SubMtx row(size_t i){
    assert(i < rows());

    double* data = &D2Idx(mData, i, 0, mRowStride, mColStride);
    return SubMtx(data, 1, cols(), mColStride);
  }
  SubMtx col(size_t i){
    assert(i < cols());

    double* data = &D2Idx(mData, 0, i, mRowStride, mColStride);
    return SubMtx(data, rows(), 1, mColStride);
  }

  /* scalar operations */
  template <typename F>
  Mtx& unary_expr(F&& f){
    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, mRowStride, mColStride) = f(D2Idx(mData, ir, ic, mRowStride, mColStride));
    return *this;
  }

  /* matrix operation */
  template <typename F>
  Mtx& binary_expr(F&& f, const Mtx& o){
    assert(rows() == o.rows() && cols() == o.cols() && rows() > 0 && cols() > 0);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, mRowStride, mColStride) = f(D2Idx(mData, ir, ic, mRowStride, mColStride),
                                                       D2Idx(o.mData, ir, ic, o.mRowStride, o.mColStride));
    return *this;
  }
  Mtx dot(const Mtx& o) const {
    assert(cols() == o.rows());

    Mtx ret = Mtx::zeros(rows(), o.cols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mRowStride, o.mColStride, mColStride, 1,
                mData, mColStride,
                o.mData, o.mColStride,
                0., ret.mData, ret.mColStride);
    return ret;
  }
  Mtx dot_plus(const Mtx& o, const Mtx& b) const {
    assert(cols() == o.rows() && cols() == b.cols());

    Mtx ret = Mtx::zeros(mRowStride, o.mColStride);
    for (size_t i = 0; i < mRowStride; ++i)
      memcpy(&ret.mData[i], b.mData, mColStride * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mRowStride, o.mColStride, mColStride, 1,
                mData, mColStride,
                o.mData, o.mColStride,
                1., ret.mData, ret.mColStride);
    return ret;
  }
  template <typename F>
  Mtx& ternary_expr(F&& f, const Mtx& x, const Mtx& y){
    assert(rows() == x.rows() && rows() == y.rows() && cols() == x.cols() && cols() == y.cols() && rows() > 0 && cols() > 0);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, mRowStride, mColStride) = f(D2Idx(mData, ir, ic, mRowStride, mColStride),
                                                       D2Idx(x.mData, ir, ic, mRowStride, mColStride),
                                                       D2Idx(y.mData, ir, ic, mRowStride, mColStride));
    return *this;
  }

  /* transpose */
  Mtx transpose() const {
    if (mRowStride == 1 || mColStride == 1){
      Mtx ret(*this);
      ret.mRows = mCols;
      ret.mCols = mRows;
      ret.mRowStride = mColStride;
      ret.mColStride = mRowStride;
      return ret;
    }
    Mtx ret(cols(), rows());
    transpose_block_SSE2x2(mData, ret.mData, mRowStride, mColStride, mColStride, mRowStride);
    return ret;
  }

  /* dimensional reduction operations */
  template <typename F>
  Mtx reduce(F&& f, MtxDim dim) const {
    switch (dim){
    case MRow: {
      Mtx ret(rows(), 1);
      for (size_t i = 0; i < rows(); ++i){
        double* data = &D2Idx(mData, i, 0, mRowStride, mColStride);
        ret(i, 0) = f(SubMtx(data, 1, cols(), mColStride));
      }
      return ret;
    } //MRow
    case MCol: {
      Mtx ret(1, cols());
      for (size_t i = 0; i < cols(); ++i){
        double* data = &D2Idx(mData, 0, i, mRowStride, mColStride);
        ret(0, i) = f(SubMtx(data, rows(), 1, mColStride));
      }
      return ret;
    } //MCol
    default: assert(false);
    } //switch
  }

  template <typename F>
  double reduce(F&& f) const {
    double ret = 0.;
    D2IterR(ir, ic, 0, rows(), 0, cols())
      ret += f(D2Idx(mData, ir, ic, mRowStride, mColStride));
    return ret;
  }

  /* statistics operations */
  std::vector<DimV> max_coeff(MtxDim dim) const {
    struct comparator_ {
      bool operator()(double a, double b){
        return a < b ? true : std::isnan(a);
      }
    } comparator;
    std::vector<DimV> ret;
    switch (dim){
    case MRow: {
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir){
        double* vmax = std::max_element(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                        &D2Idx(mData, ir, cols(), mRowStride, mColStride),
                                        comparator);
        size_t dist = std::distance(&D2Idx(mData, ir, 0, mRowStride, mColStride), vmax);
        if (std::isnan(*vmax)) ret.emplace_back((DimV){std::numeric_limits<size_t>::max(), nan("")});
        else                   ret.emplace_back((DimV){dist, *vmax});
      }
    } break; //mRow
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        size_t dist = std::numeric_limits<size_t>::max();
        double vmax = std::numeric_limits<double>::min();
        for (size_t ir = 0; ir < rows(); ++ir)
          if (not std::isnan(D2Idx(mData, ir, ic, mRowStride, mColStride)) &&
              D2Idx(mData, ir, ic, mRowStride, mColStride) > vmax){
            vmax = D2Idx(mData, ir, ic, mRowStride, mColStride);
            dist = ir;
          }
        if (dist == std::numeric_limits<size_t>::max()) ret.emplace_back((DimV){dist, nan("")});
        else                                            ret.emplace_back((DimV){dist, vmax});
      }
    } break; //mCol
    default: assert(false);
    } //switch
    return ret;
  }

  DimV max_coeff() const {
    struct comparator_ {
      bool operator()(double a, double b){
        return a < b ? true : std::isnan(a);
      }
    } comparator;

    DimV ret;
    ret.idx = std::numeric_limits<size_t>::max();
    ret.val = std::numeric_limits<double>::min();
    for (size_t ir = 0; ir < rows(); ++ir){
      double* pvmax = std::max_element(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                       &D2Idx(mData, ir, cols(), mRowStride, mColStride),
                                       comparator);
      if (*pvmax > ret.val){
        ret.idx = std::distance(&D2Idx(mData, ir, 0, mRowStride, mColStride), pvmax) + ir * cols();
        ret.val = *pvmax;
      }
    }

    if (ret.idx == std::numeric_limits<size_t>::max())
      ret.val = nan("");
    return ret;
  }

  std::vector<DimV> min_coeff(MtxDim dim) const {
    struct comparator_ {
      bool operator()(double a, double b){
        return a < b ? true : std::isnan(b);
      }
    } comparator;
    std::vector<DimV> ret;
    switch (dim){
    case MRow: {
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir){
        double* vmin = std::min_element(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                        &D2Idx(mData, ir, cols(), mRowStride, mColStride),
                                        comparator);
        size_t  dist = std::distance(&D2Idx(mData, ir, 0, mRowStride, mColStride), vmin);
        if (std::isnan(*vmin)) ret.emplace_back((DimV){std::numeric_limits<size_t>::max(), nan("")});
        else                   ret.emplace_back((DimV){dist, *vmin});
      }
    } break; //MRow
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        size_t dist = std::numeric_limits<size_t>::max();
        double vmin = std::numeric_limits<double>::max();
        for (size_t ir = 0; ir < rows(); ++ir)
          if (not std::isnan(D2Idx(mData, ir, ic, mRowStride, mColStride)) &&
              D2Idx(mData, ir, ic, mRowStride, mColStride) < vmin){
            vmin = D2Idx(mData, ir, ic, mRowStride, mColStride);
            dist = ir;
          }
        if (dist == std::numeric_limits<size_t>::max()) ret.emplace_back((DimV){dist, nan("")});
        else                                            ret.emplace_back((DimV){dist, vmin});
      }
    } break; //MCol
    default: assert(false);
    } //switch
    return ret;
  }

  DimV min_coeff() const {
    struct comparator_ {
      bool operator()(double a, double b){
        return a < b ? true : std::isnan(b);
      }
    } comparator;

    DimV ret;
    ret.idx = std::numeric_limits<size_t>::max();
    ret.val = std::numeric_limits<double>::max();
    for (size_t ir = 0; ir < rows(); ++ir){
      double *pvmin = std::min_element(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                       &D2Idx(mData, ir, cols(), mRowStride, mColStride),
                                       comparator);
      if (*pvmin < ret.val){
        ret.idx = std::distance(&D2Idx(mData, ir, 0, mRowStride, mColStride), pvmin) + ir * cols();
        ret.val = *pvmin;
      }
    }
    if (ret.idx == std::numeric_limits<size_t>::max())
      ret.val = nan("");
    return ret;
  }

  std::vector<double> sum(MtxDim dim) const {
    struct binop_ {
      double operator()(double a, double b){
        if (std::isnan(b)) return a;
        else               return a + b;
      }
    } binop;
    std::vector<double> ret;
    switch (dim){
    case MRow:
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir)
        ret.push_back(std::accumulate(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                      &D2Idx(mData, ir, cols(), mRowStride, mColStride),
                                      0., binop));
    break; //MRow
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        double sum = 0.;
        for (size_t ir = 0; ir < rows(); ++ir)
          if (not std::isnan(D2Idx(mData, ir, ic, mRowStride, mColStride)))
            sum += D2Idx(mData, ir, ic, mRowStride, mColStride);
        ret.push_back(sum);
      }
    } break; //MCol
    default: assert(false);
    } // switch
    return ret;
  }

  double sum() const {
    struct binop_ {
      double operator()(double a, double b){
        if (std::isnan(b)) return a;
        else               return a + b;
      }
    } binop;

    return std::accumulate(mData, &mData[mRowStride * mColStride], 0., binop);
  }

  std::vector<double> mean(MtxDim dim) const {
    struct nan_compare_ {
      bool operator()(double d){
        return std::isnan(d);
      }
    } nan_compare;
    std::vector<double> sums = sum(dim);
    std::vector<double> sizes;
    sizes.reserve(sums.size());
    switch (dim){
    case MRow: {
      for (size_t ir = 0; ir < rows(); ++ir){
        size_t na = std::count_if(&D2Idx(mData, ir, 0, mRowStride, mColStride),
                                  &D2Idx(mData, ir, cols(), mRowStride, mColStride), nan_compare);
        sizes.push_back(cols() - na);
      }
    } break; //MRow
    case MCol: {
      for (size_t ic = 0; ic < cols(); ++ic){
        size_t na = 0;
        for (size_t ir = 0; ir < rows(); ++ir)
          if (std::isnan(D2Idx(mData, ir, ic, mRowStride, mColStride)))
              na++;
        sizes.push_back(rows() - na);
      }
    } break; //MCol
    default: assert(false);
    }

    for (size_t i = 0; i < sums.size(); ++i)
      sums[i] /= sizes[i];
    return sums;
  }

  double mean() const {
    struct nan_compare_ {
      bool operator()(double d){
        return std::isnan(d);
      }
    } nan_compare;

    double s = sum();
    size_t na = std::count_if(mData, &mData[mRowStride * mColStride], nan_compare);

    return s / (double)(rows() * cols() - na);
  }

  /* save */
  std::ostream& save(std::ostream& out) const {
    assert(mData != nullptr);
    out.write((char*)&mRows, sizeof(size_t));
    out.write((char*)&mCols, sizeof(size_t));
    out.write((char*)mData, sizeof(double) * mRowStride * mColStride);
    return out;
  }
  void save(const char* filename){
    std::ofstream out(filename, std::ofstream::out);

    assert(out.good());
    save(out);

    out.close();
  }

  /* implicit conversion to SubMtx */
  operator SubMtx() {
    return SubMtx(mData, mRows, mCols, mColStride);
  }

  /* static generators */
  static Mtx zeros(size_t r, size_t c){
    return Mtx(r, c, 0.);
  }
  template <typename DIST = std::normal_distribution<double>>
  static Mtx random(size_t r, size_t c, DIST distribution = std::normal_distribution<double>(0.0, 0.5)){
    Mtx ret(r, c);
    std::default_random_engine& eng = get_default_random_engine();
    ret.unary_expr([&distribution, &eng](double){ return distribution(eng); });
    return ret;
  }
  static Mtx ident(size_t s){
    Mtx ret(s, s, 0.);
    size_t us = round_up(s);
    D1Iter(i, 0, s) D2Idx(ret.mData, i, i, us, us) = 1.;
    return ret;
  }
};

auto addition_operation_    = [](double i, double j){ return i + j; };
auto subtraction_operation_ = [](double i, double j){ return i - j; };

/* matrix scalar operations */
Mtx operator+(const Mtx& a, double b){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return v + b; });
  return ret;
}
Mtx operator+(double b, const Mtx& a){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return v + b; });
  return ret;
}
Mtx operator-(const Mtx& a, double b){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return v - b; });
  return ret;
}
Mtx operator-(double b, const Mtx& a){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return b - v; });
  return ret;
}
Mtx operator*(const Mtx& a, double b){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return v * b; });
  return ret;
}
Mtx operator*(double b, const Mtx& a){
  Mtx ret(a);
  ret.unary_expr([b](double v){ return v * b; });
  return ret;
}
Mtx operator/(const Mtx& a, double b){
  Mtx ret(a);
  ret.unary_expr([b](double v){
      if (b == 0.) return nan("");
      else         return v / b;
  });
  return ret;
}
Mtx operator/(double b, const Mtx& a){
  Mtx ret(a);
  ret.unary_expr([b](double v){
      if (v == 0.) return nan("");
      else         return b / v;
  });
  return ret;
}

/* binary matrix operations */
Mtx operator+(const Mtx& a, const Mtx& b){
  Mtx ret(a);
  return ret.binary_expr(addition_operation_, b);
}
Mtx operator-(const Mtx& a, const Mtx& b){
  Mtx ret(a);
  return ret.binary_expr(subtraction_operation_, b);
}
Mtx operator*(const Mtx& a, const Mtx& b){
  return a.dot(b);
}

/* matrix scalar assignment operations */
Mtx& operator+=(Mtx& a, double b){
  a.unary_expr([b](double v){ return v + b; });
  return a;
}
Mtx& operator-=(Mtx& a, double b){
  a.unary_expr([b](double v){ return v - b; });
  return a;
}
Mtx& operator*=(Mtx& a, double b){
  a.unary_expr([b](double v){ return v * b; });
  return a;
}
Mtx& operator/=(Mtx& a, double b){
  a.unary_expr([b](double v){
      if (b == 0.) return nan("");
      else         return v / b;
  });
  return a;
}

/* matrix binary assignment operations */
Mtx& operator+=(Mtx& a, const Mtx& b){
  return a.binary_expr(addition_operation_, b);
}
Mtx& operator-=(Mtx& a, const Mtx& b){
  return a.binary_expr(subtraction_operation_, b);
}

/* export operations */
std::ostream& operator << (std::ostream& out, const Mtx& m){
  for (size_t i = 0; i < m.rows(); ++i){
    bool is_first = true;
    for (size_t j = 0; j < m.cols(); ++j){
      if (is_first) is_first = false;
      else          out << " ";
      out << m(i, j);
    }
    out << std::endl;
  }
  return out;
}

/* comparison operations */
bool compare_double(double a, double b){
  return a == b || std::abs(a - b) < std::abs(std::min(a, b)) * std::numeric_limits<double>::epsilon();
}
bool operator==(const Mtx& a, const Mtx& b){
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;

  for (size_t i = 0; i < a.rows(); ++i)
    for (size_t j = 0; j < a.cols(); ++j)
      if (not compare_double(a(i, j), b(i, j))){
        return false;
      }
  return true;
}
bool operator!=(const Mtx& a, const Mtx& b){
  return not operator==(a, b);
}

} //ML

#endif //ML_MATRIX

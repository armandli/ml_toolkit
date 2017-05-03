#ifndef ML_MATRIX
#define ML_MATRIX

//TODO: make matrix NaN safe
//TODO: SubMtx copy may expand as a grid against the other or taking a segment of the other?
//TODO: add cublas usages
//TODO: add sum of entire matrix
//TODO: add print of SubMtx

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

#include <cblas.h>

#ifdef __NVCC__
#include <cublas_v2.h>
#include <curand.h>

#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {   
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }   
}
#endif //__NVCC__

#include <ml_common.h>

namespace ML {

enum MtxDim {
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
  size_t  mSlide;

public:
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }

  SubMtx(double* data, size_t r, size_t c, size_t s):
    mData(data), mRows(r), mCols(c), mSlide(s){}
  SubMtx(const SubMtx& o){
    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mSlide) = D2Idx(o.mData, ir, ic, o.rows(), o.mSlide);
  }
  SubMtx(SubMtx&& o){
    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mSlide) = D2Idx(o.mData, ir, ic, o.rows(), o.mSlide);
  }
  ~SubMtx() = default;

  SubMtx& operator=(const SubMtx& o){
    if (this == &o) return *this;

    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mSlide) = D2Idx(o.mData, ir, ic, o.rows(), o.mSlide);
    return *this;
  }
  SubMtx& operator=(SubMtx&& o){
    if (this == &o) return *this;

    assert(rows() == o.rows() && cols() == o.cols());
    assert(mData != nullptr && o.mData != nullptr);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), mSlide) = D2Idx(o.mData, ir, ic, o.rows(), o.mSlide);
    return *this;
  }
};

class Mtx {
  double* mData;
  size_t  mRows;
  size_t  mCols;
#ifdef __NVCC__
  double* mCada;
#endif

public:
  Mtx(): mData(nullptr), mRows(0), mCols(0) {}
  Mtx(size_t r, size_t c, double def = 0.):
    mData(new double[r * c]), mRows(r), mCols(c) {
    for (size_t i = 0; i < mRows * mCols; ++i)
      mData[i] = def;
  }
  explicit Mtx(std::istream& in) {
    size_t r, c; in >> r >> c;
    mRows = r;
    mCols = c;
    mData = new double[r * c];
    //TODO: wrong
    for (size_t i = 0; i < r * c; ++i){
      std::string bytes;
      in >> bytes;
      assert(bytes.length() == sizeof(double));
      memcpy(&mData[i], bytes.c_str(), sizeof(double));
    }
  }
  Mtx(size_t r, size_t c, const std::vector<double>& data):
    mData(new double[r * c]), mRows(r), mCols(c) {
    assert(data.size() >= r * c);

    memcpy(mData, data.data(), r * c * sizeof(double));
  }
  Mtx(const Mtx& o):
    mData(new double[o.rows() * o.cols()]), mRows(o.rows()), mCols(o.cols()){
    memcpy(mData, o.mData, sizeof(double) * o.rows() * o.cols());
  }
  Mtx(Mtx&& o):
    mData(o.mData), mRows(o.rows()), mCols(o.cols()) {
    o.mData = nullptr;
    o.mRows = o.mCols = 0;
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
      mData = new double[o.rows() * o.cols()];
      memcpy(mData, o.mData, o.rows() * o.cols() * sizeof(double));
    } else if (o.mData == nullptr && mData != nullptr){
      delete[] mData;
      mData = nullptr;
    } else if (rows() * cols() != o.rows() * o.cols()){
      delete[] mData;
      mData = new double[o.rows() * o.cols()];
      memcpy(mData, o.mData, o.rows() * o.cols() * sizeof(double));
    } else {
      memcpy(mData, o.mData, o.rows() * o.cols() * sizeof(double));
    }
    mRows = o.rows();
    mCols = o.cols();
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
    o.mData = nullptr;
    o.mRows = o.mCols = 0;
    return *this;
  }

  /* accessors */
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }
  double operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());

    return D2Idx(mData, i, j, rows(), cols());
  }
  SubMtx block(size_t rp, size_t cp, size_t rz = 1, size_t cz = 1) {
    assert(rp < rows() && cp < cols() && rp + rz <= rows() && cp + cz <= cols());

    double* data = &D2Idx(mData, rp, cp, rows(), cols());
    return SubMtx(data, rz, cz, cols());
  }

  /* scalar operations */
  template <typename F>
  Mtx& unary_expr(F&& f){
    D1Iter(i, 0, rows() * cols()) mData[i] = f(mData[i]);
    return *this;
  }
  template <typename F>
  Mtx unary_expr(F&& f) const {
    Mtx ret(*this);
    D1Iter(i, 0, rows() * cols()) ret.mData[i] = f(ret.mData[i]);
    return ret;
  }

  /* matrix operation */
  template <typename F>
  Mtx& binary_expr(F&& f, const Mtx& o){
    assert(rows() == o.rows() && cols() == o.cols() && rows() > 0 && cols() > 0);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(mData, ir, ic, rows(), cols()) = f(D2Idx(mData, ir, ic, rows(), cols()), D2Idx(o.mData, ir, ic, rows(), cols()));
    return *this;
  }
  template <typename F>
  Mtx binary_expr(F&& f, const Mtx& o) const {
    assert(rows() == o.rows() && cols() == o.cols() && rows() > 0 && cols() > 0);
    Mtx ret(*this);

    D2IterR(ir, ic, 0, rows(), 0, cols())
      D2Idx(ret.mData, ir, ic, rows(), cols()) = f(D2Idx(ret.mData, ir, ic, rows(), cols()), D2Idx(o.mData, ir, ic, rows(), cols()));
    return ret;
  }
  Mtx dot(const Mtx& o) const {
    assert(cols() == o.rows());

    Mtx ret = Mtx::zeros(rows(), o.cols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows(), o.cols(), cols(), 1,
                mData, cols(),
                o.mData, o.cols(),
                0., ret.mData, ret.cols());
    return ret;
  }

  /* transpose */
  void transpose(){
    mRows ^= mCols;
    mCols ^= mRows;
    mRows ^= mCols;

    //this is a columar matrix based transpose algorithm
    std::vector<bool> visited(rows() * cols(), false);
    for (size_t i = 0; i < cols(); ++i)
      for (size_t j = 0; j < rows(); ++j){
        size_t orig = j + i * rows();
        size_t dest = i + j * cols();

        if (visited[orig]) continue;
        if (orig == dest) continue;

        double valence = mData[dest];
        while (not visited[orig]){
          mData[dest] = mData[orig];
          visited[dest] = true;

          size_t col = orig / cols();
          size_t row = orig % cols();
          dest = orig;
          orig = col + row * rows();
        }
        mData[dest] = valence;
        visited[dest] = true;
      }
  }

  /* statistics operations */
  std::vector<DimV> max_coeff(MtxDim dim){
    std::vector<DimV> ret;
    switch (dim){
    case MRow: {
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir){
        double* vmax = std::max_element(&mData[ir * cols()], &mData[(ir + 1) * cols()]);
        size_t dist  = std::distance(&mData[ir * cols()], vmax);
        ret.emplace_back((DimV){dist, *vmax});
      }
    } break; //mRow
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        size_t dist = 0;
        double vmax = std::numeric_limits<double>::min(), tmp;
        for (size_t ir = 0; ir < rows(); ++ir)
          if ((tmp = D2Idx(mData, ir, ic, rows(), cols())) > vmax){
            vmax = tmp;
            dist = ir;
          }
        ret.emplace_back((DimV){dist, vmax});
      }
    } break; //mCol
    default: assert(false);
    } //switch
    return ret;
  }
  std::vector<DimV> min_coeff(MtxDim dim){
    std::vector<DimV> ret;
    switch (dim){
    case MRow: {
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir){
        double* vmax = std::min_element(&mData[ir * cols()], &mData[(ir + 1) * cols()]);
        size_t dist  = std::distance(&mData[ir * cols()], vmax);
        ret.emplace_back((DimV){dist, *vmax});
      }
    } break; //mRow
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        size_t dist = 0;
        double vmax = std::numeric_limits<double>::max(), tmp;
        for (size_t ir = 0; ir < rows(); ++ir)
          if ((tmp = D2Idx(mData, ir, ic, rows(), cols())) < vmax){
            vmax = tmp;
            dist = ir;
          }
        ret.emplace_back((DimV){dist, vmax});
      }
    } break; //mCol
    default: assert(false);
    } //switch
    return ret;
  }
  std::vector<double> sum(MtxDim dim){
    //TODO: how to deal with NaN here properly
    std::vector<double> ret;
    switch (dim){
    case MRow: {
      ret.reserve(rows());
      for (size_t ir = 0; ir < rows(); ++ir)
        ret.emplace_back(std::accumulate(&mData[ir * cols()], &mData[(ir + 1) * cols()], 0.));
    } break;
    case MCol: {
      ret.reserve(cols());
      for (size_t ic = 0; ic < cols(); ++ic){
        double accu = 0.;
        for (size_t ir = 0; ir < rows(); ++ir)
          accu += D2Idx(mData, ir, ic, rows(), cols());
        ret.emplace_back(accu);
      }
    } break;
    default: assert(false);
    } // switch
    return ret;
  }
  std::vector<double> mean(MtxDim dim){
    std::vector<double> sums = sum(dim);
    double sz;
    switch (dim){
    case MRow: sz = (double)rows(); break;
    case MCol: sz = (double)cols(); break;
    default: assert(false);
    } //switch
    std::for_each(sums.begin(), sums.end(), [sz](double& v){ v /= sz; });
    return sums;
  }

  /* save */
  std::ostream& save(std::ostream& out) const {
    assert(mData != nullptr);
    out << rows() << " " << cols() << " ";
    std::for_each(mData, &mData[rows() * cols()], [&out](double& v){
        char bytes[sizeof(double) + 1];
        memcpy(bytes, &v, sizeof(double));
        bytes[sizeof(double)] = 0;
        out << bytes;
    });
    return out;
  }

  /* implicit conversion to SubMtx */
  operator SubMtx() {
    return SubMtx(mData, mRows, mCols, mCols);
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
    D1Iter(i, 0, s) D2Idx(ret.mData, i, i, s, s) = 1.;
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
  return a.binary_expr(addition_operation_, b);
}
Mtx operator-(const Mtx& a, const Mtx& b){
  return a.binary_expr(subtraction_operation_, b);
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

  bool is_equal = true;
  a.binary_expr([&is_equal](double i, double j){
      if (not compare_double(i, j))
        is_equal = false;
      return i;
  }, b);
  return is_equal;
}
bool operator!=(const Mtx& a, const Mtx& b){
  return not operator==(a, b);
}

} //ML

#endif //ML_MATRIX

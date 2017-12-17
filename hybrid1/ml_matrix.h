#ifndef ML_MATRIX
#define ML_MATRIX

#include <cassert>
#include <cstdlib>
#include <fstream>

#include <ml_common.h>
#include <ml_mtxop.h>
#include <ml_sse.h>
#include <ml_exprtree.h>
#include <ml_instr.h>
#include <ml_ssa_decl.h>

namespace ML {

class MemInstrContext;

class Memory {
  double* mData;
  size_t  mRows;
  size_t  mCols;
  mutable SSA mSSA;

  friend struct ComputeMtxCommunicator;
protected:
  double* data() const { return mData; }
  void set_data(double* data){ mData = data; }
  void delete_data(){
    if (mData) free(mData);
    mData = nullptr;
  }

  bool is_ssa() const { return not mSSA.empty(); }
  bool is_ssa_empty() const { return mSSA.empty(); }
  SSA& ssa() const { return mSSA; }
  void clear_ssa() const { mSSA.clear(); }

public:
  explicit Memory(double* data): mData(data), mRows(0), mCols(0) {}
  Memory(double* data, size_t rows, size_t cols): mData(data), mRows(rows), mCols(cols) {}
  ~Memory() = default;

  size_t rows() const {
    return mRows;
  }

  void set_rows(size_t r){
    mRows = r;
  }

  size_t cols() const {
    return mCols;
  }

  void set_cols(size_t c){
    mCols = c;
  }
};

class Mtx: public Memory {
  size_t  mRowStride;
  size_t  mColStride;

  /* load matrix from fstream */
  void load(std::istream& in){
    assert(in.good());

    size_t a, b;
    in.read((char*)&a, sizeof(size_t));
    in.read((char*)&b, sizeof(size_t));
    set_rows(a);
    set_cols(b);

    assert(rows() > 0 && cols() > 0);

    mRowStride = roundup_row(rows());
    mColStride = roundup_col(cols());

    double* ptr = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * mRowStride * mColStride);
    set_data(ptr);

    in.read((char*)data(), sizeof(double) * mRowStride * mColStride);
  }

  friend struct SSAMtxCommunicator;
public:
  //should not use these functions
  Mtx(const Mtx&): Memory(nullptr) { assert(false); }
  Mtx& operator=(const Mtx&){ assert(false); }

  Mtx(): Memory(nullptr), mRowStride(0), mColStride(0) {}
  Mtx(size_t r, size_t c, double v = 0.):
    Memory(nullptr, r, c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    double* ptr = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * mRowStride * mColStride);
    set_data(ptr);
    if (v == 0.) memset(data(), 0, sizeof(double) * mRowStride * mColStride);
    else         SSE::const_init_1d_sse_pd(data(), v, rows(), mColStride);
  }
  Mtx(size_t r, size_t c, const std::vector<double>& v):
    Memory(nullptr, r, c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    assert(v.size() >= r * c);
    double* ptr = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * mRowStride * mColStride);
    set_data(ptr);
    //TODO: use SSE instructions to do this
    for (size_t ir = 0, k = 0; ir < r; ++ir, ++k)
      for (size_t ic = 0; ic < c; ++ic)
        data()[ir * mColStride + ic] = v[k];
  }
  Mtx(size_t r, size_t c, RandomizationType rtype, double p1, double p2):
    Memory(nullptr, r, c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    double* ptr = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * mRowStride * mColStride);
    set_data(ptr);
    switch (rtype){
      case RandomizationType::Uniform:  MTXOP::rnd_uniform_init_2d_mtxop_pd(data(), p1, p2, rows(), cols(), mColStride); break;
      case RandomizationType::Gaussian: MTXOP::rnd_normal_init_2d_mtxop_pd(data(), p1, p2, rows(), cols(), mColStride); break;
      default: assert(false);
    }
  }
  explicit Mtx(std::istream& in): Memory(nullptr) {
    load(in);
  }
  explicit Mtx(const char* filename): Memory(nullptr) {
    std::ifstream in(filename, std::ifstream::in);
    assert(in.good());
    load(in);
    in.close();
  }
  template <typename CRTP>
  Mtx(MtxBase<CRTP>&& expr):
    Memory(nullptr), mRowStride(0), mColStride(0) {
    ssa() = to_ssa(expr, *this);
  }

  ~Mtx(){ delete_data(); }

  template <typename CRTP>
  Mtx& operator=(MtxBase<CRTP>&& expr){
    ssa() = to_ssa(expr, *this);
    return *this;
  }

  //TODO: is this even something that's good?
  void init(size_t r, size_t c){
    set_rows(r);
    set_cols(c);
    mRowStride = roundup_row(r);
    mColStride = roundup_col(c);
    double* ptr = (double*)aligned_alloc(ALIGNMENT_WIDTH, sizeof(double) * mRowStride * mColStride);
    set_data(ptr);
  }

  void evaluate(MemArena& arena){
    if (is_ssa_empty()) return;
    memvaluateSSA(ssa(), arena);
  }
  void evaluate(CUDA::CUDArena&){
    //TODO
  }

  /* accessors */
  size_t rowstride() const { return mRowStride; }
  size_t colstride() const { return mColStride; }
  double operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());
    assert(is_ssa_empty());

    return D2Idx(data(), i, j, mRowStride, mColStride);
  }
  double& operator()(size_t i, size_t j){
    assert(i < rows() && j < cols());
    assert(is_ssa_empty());

    return D2Idx(data(), i, j, mRowStride, mColStride);
  }
};

std::ostream& operator << (std::ostream& out, const Mtx& m){
  for (size_t ir = 0; ir < m.rows(); ++ir){
    for (size_t ic = 0; ic < m.cols(); ++ic)
      out << m(ir, ic) << " ";
    out << "\n";
  }
  return out;
}

class ReductionResult: public Memory {
  double mVal;
public:
  ReductionResult(): Memory(&mVal, 1, 1), mVal(0.) {}

  template <typename CRTP>
  ReductionResult(MtxBase<CRTP>&& expr): Memory(&mVal, 1, 1), mVal(0.) {
    ssa() = to_ssa(expr, *this);
  }
  ~ReductionResult() = default;

  template <typename CRTP>
  ReductionResult& operator=(MtxBase<CRTP>&& expr){
    assert(is_ssa_empty());
    ssa() = to_ssa(expr, *this);
    return *this;
  }

  void evaluate(MemArena& arena){
    if (is_ssa_empty()) return;
    memvaluateSSA(ssa(), arena);
  }

  void evaluate(CUDA::CUDArena&){
    //TODO
  }

  operator double() const {
    assert(is_ssa_empty());
    return mVal;
  }
};

} // ML

#endif//ML_MATRIX

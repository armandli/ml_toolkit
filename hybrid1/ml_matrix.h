#ifndef ML_MATRIX
#define ML_MATRIX

#include <cassert>
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

  friend class ComputeMtxCommunicator;
protected:
  double* data() const { return mData; }
  void set_data(double* data){ mData = data; }
  void delete_data(){
    if (mData) delete[] mData;
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

    set_data(new double[mRowStride * mColStride]);

    in.read((char*)data(), sizeof(double) * mRowStride * mColStride);
  }

  friend class SSAMtxCommunicator;
public:
  Mtx(): Memory(nullptr), mRowStride(0), mColStride(0) {}
  Mtx(size_t r, size_t c, double v = 0.):
    Memory(nullptr, r, c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    set_data(new double[mRowStride * mColStride]);
    SSE::const_init_2d_sse_pd(data(), v, rows(), cols(), mRowStride, mColStride);
  }
  Mtx(size_t r, size_t c, RandomizationType rtype, double p1, double p2):
    Memory(nullptr, r, c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    set_data(new double[mRowStride * mColStride]);
    switch (rtype){
      case RandomizationType::Uniform:  MTXOP::rnd_uniform_init_2d_mtxop_pd(data(), p1, p2, rows(), cols(), mRowStride, mColStride); break;
      case RandomizationType::Gaussian: MTXOP::rnd_normal_init_2d_mtxop_pd(data(), p1, p2, rows(), cols(), mRowStride, mColStride); break;
      default: assert(false);
    }
  }
  Mtx(std::istream& in): Memory(nullptr) {
    load(in);
  }
  Mtx(const char* filename): Memory(nullptr) {
    std::ifstream in(filename, std::ifstream::in);
    assert(in.good());
    load(in);
    in.close();
  }
  template <typename CRTP>
  Mtx(MtxBase<CRTP>&& expr):
    Memory(nullptr), mRowStride(0), mColStride(0) {
    //to_ssa is friend, and modifies this matrix's data
    ssa() = to_ssa(expr, *this);
  }

  ~Mtx(){ delete_data(); }

  template <typename CRTP>
  Mtx& operator=(MtxBase<CRTP>&& expr){
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

  /* accessors */
//  size_t rows() const { return mRows; }
//  size_t cols() const { return mCols; }
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

class ReductionResult: public Memory {
  double mVal;
public:
  template <typename CRTP>
  ReductionResult(MtxBase<CRTP>&& expr): Memory(&mVal, 1, 1), mVal(0) {
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

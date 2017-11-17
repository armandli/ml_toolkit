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

class Mtx {
  double* mData;
  size_t  mRows;
  size_t  mCols;
  size_t  mRowStride;
  size_t  mColStride;
  mutable SSA   mSSA;

  /* load matrix from fstream */
  void load(std::istream& in){
    assert(in.good());

    in.read((char*)&mRows, sizeof(size_t));
    in.read((char*)&mCols, sizeof(size_t));

    assert(mRows > 0 && mCols > 0);

    mRowStride = roundup_row(mRows);
    mColStride = roundup_col(mCols);

    mData = new double[mRowStride * mColStride];

    in.read((char*)mData, sizeof(double) * mRowStride * mColStride);
  }

  bool is_ssa() const {
    return mSSA.instructions.size() != 0;
  }

  double* data() const {
    return mData;
  }

  SSA& ssa() const {
    return mSSA;
  }

  void clear_ssa() const {
    mSSA.clear();
  }

  friend class SSAMtxCommunicator;
  friend class ComputeMtxCommunicator;
public:
  Mtx(): mData(nullptr), mRows(0), mCols(0), mRowStride(0), mColStride(0) {}
  Mtx(size_t r, size_t c, double v = 0.):
    mData(nullptr), mRows(r), mCols(c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    mData = new double[mRowStride * mColStride];
    SSE::const_init_2d_sse_pd(mData, v, mRows, mCols, mRowStride, mColStride);
  }
  Mtx(size_t r, size_t c, RandomizationType rtype, double p1, double p2):
    mData(nullptr), mRows(r), mCols(c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)) {
    mData = new double[mRowStride * mColStride];
    switch (rtype){
      case RandomizationType::Uniform:  MTXOP::rnd_uniform_init_2d_mtxop_pd(mData, p1, p2, mRows, mCols, mRowStride, mColStride); break;
      case RandomizationType::Gaussian: MTXOP::rnd_normal_init_2d_mtxop_pd(mData, p1, p2, mRows, mCols, mRowStride, mColStride); break;
      default: assert(false);
    }
  }
  Mtx(std::istream& in){
    load(in);
  }
  Mtx(const char* filename){
    std::ifstream in(filename, std::ifstream::in);
    assert(in.good());
    load(in);
    in.close();
  }
  template <typename CRTP>
  Mtx(MtxBase<CRTP>&& expr):
    mData(nullptr), mRows(0), mCols(0), mRowStride(0), mColStride(0) {
    //to_ssa is friend, and modifies this matrix's data
    mSSA = to_ssa(expr, *this);
  }

  ~Mtx(){
    if (mData){
      delete[] mData;
      mData = nullptr;
    }
  }

  template <typename CRTP>
  Mtx& operator=(MtxBase<CRTP>&& expr){
    assert(mSSA.empty());
    mSSA = to_ssa(expr, *this);
    return *this;
  }

  void evaluate(MemArena& arena){
    if (mSSA.empty()) return;

    memvaluateSSA(mSSA, arena);
  }
  void evaluate(CUDA::CUDArena&){
    //TODO
  }

  /* accessors */
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }
  size_t rowstride() const { return mRowStride; }
  size_t colstride() const { return mColStride; }
  double operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());
    assert(mSSA.empty());

    return D2Idx(mData, i, j, mRowStride, mColStride);
  }
  double& operator()(size_t i, size_t j){
    assert(i < rows() && j < cols());
    assert(mSSA.empty());

    return D2Idx(mData, i, j, mRowStride, mColStride);
  }

  //TODO
};

} // ML

#endif//ML_MATRIX

#ifndef NN_MATRIX
#define NN_MATRIX

#include <cstddef>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

#define D1Iter(i, ib, ie) \
  for (size_t i = ib; i < ie; ++i)
#define D2IterC(ir, ic, rb, re, cb, ce) \
  for (size_t ic = cb; ic < ce; ++ic) for (size_t ir = rb; ir < re; ++ir)
#define D2IterR(ir, ic, rb, re, cb, ce) \
  for (size_t ir = rb; ir < re; ++ir) for (size_t ic = cb; ic < ce; ++ic)

const unsigned long long PTR_MASK = 0xFFFFFFFFFFFFFFFEUL;
const unsigned STRIDE = 32 * 1024; //TODO

template <typename T> class RowRef;
template <typename T> class ColRef;

enum MtxDim {
  MCol,
  MRow,
};

/* column locality matrix */
template <typename T>
class Mtx {
  friend class RowRef<T>;
  friend class ColRef<T>;

  mutable T* mData;
  size_t mRows;
  size_t mCols;

  using IndexFP = size_t (Mtx::*)(size_t, size_t) const;

  size_t index(size_t r, size_t c) const {
    return r + c * mRows;
  }
  size_t rindex(size_t r, size_t c) const {
    return c + r * mCols;
  }
  T* dataptr() const {
    return (T*)(((size_t)mData) & PTR_MASK);
  }
  bool is_rotated() const {
    return ((size_t)mData & 0x1UL);
  }
  void flip_bit() const {
    mData = (T*)((size_t)mData ^ 0x1UL);
  }
  
  void rotate() const {
    T* data = dataptr();
    vector<bool> visited(rows() * cols(), false);
    for (size_t i = 0; i < cols(); ++i)
      for (size_t j = 0; j < rows(); ++j){
        size_t orig = j + i * rows();
        size_t dest = i + j * cols();

        if (visited[orig]) continue;
        if (orig == dest) continue;

        T valence = data[dest];
        while (not visited[orig]){
          data[dest] = data[orig];
          visited[dest] = true;

          size_t col = orig / cols();
          size_t row = orig % cols();
          dest = orig;
          orig = col + row * rows();
        }
        data[dest] = valence;
        visited[dest] = true;
      }
  }
public:
  /* constructors, destructors, assignment operators */
  Mtx(): mData(nullptr) {} //mRows mCols undefined
  Mtx(size_t r, size_t c, T s = 0): mData(nullptr), mRows(r), mCols(c) {
    mData = new T[r * c];
    D1Iter(i, 0, r * c) mData[i] = s;
  }
  //NOTE: vector<T> is a vector of columns
  Mtx(size_t r, size_t c, const vector<T>& data): mData(nullptr), mRows(r), mCols(c) {
    assert(data.size() == r * c);
    mData = new T[r * c];
    memcpy(mData, data.data(), sizeof(T) * r * c);
  }
  Mtx(const Mtx& o): mRows(o.mRows), mCols(o.mCols) {
    T* optr = o.dataptr();
    if (not optr) return;
    mData = new T[rows() * cols()];
    memcpy(dataptr(), optr, sizeof(T) * rows() * cols());
    if (o.is_rotated())
      flip_bit();
  }
  Mtx(Mtx&& o) = default;
  ~Mtx() {
    T* data = dataptr();
    if (data) delete[] data;
    mData = nullptr;
  }
  Mtx& operator=(const Mtx& o){
    T* data = dataptr();
    T* odata = o.dataptr();
    if (&o == this) return *this;
    delete[] data;

    mData = new T[o.rows() * o.cols()];
    mRows = o.rows();
    mCols = o.cols();
    memcpy(mData, odata, sizeof(T) * rows() * cols());
    if (o.is_rotated())
      flip_bit();
    return *this;
  }
  Mtx& operator=(Mtx&& o){
    if (this == &o) return *this;

    T* data = dataptr();
    delete[] data;

    mData = o.mData;
    mRows = o.mRows;
    mCols = o.mCols;

    o.mData = nullptr;
    return *this;
  }
  explicit Mtx(const string& filename): mData(nullptr) {
    ifstream in(filename.c_str());
    if (not in.is_open()){
      cout << "could not open input file " << filename << " to load" << endl;
      return;
    }

    size_t r, c; in >> r >> c;

    vector<T> vec; T v;
    while (in >> v) vec.push_back(v);

    mRows = r;
    mCols = c;

    mData = new T[vec.size()];

    memcpy(mData, vec.data(), sizeof(T) * r * c);
  }

  /* accessors */
  size_t rows() const { return mRows; }
  size_t cols() const { return mCols; }
  const RowRef<T> row(size_t idx) const;
  RowRef<T> row(size_t idx);
  const ColRef<T> col(size_t idx) const;
  ColRef<T> col(size_t idx);
  T operator()(size_t r, size_t c) const {
    assert(dataptr() != nullptr);
    if (not is_rotated()) return dataptr()[index(r, c)];
    else                  return dataptr()[rindex(r, c)];
  }
  T& operator()(size_t r, size_t c){
    assert(dataptr() != nullptr);
    if (not is_rotated()) return dataptr()[index(r, c)];
    else                  return dataptr()[rindex(r, c)];
  }

  /* scalar operations */
  Mtx& add(T val){
    T* data = dataptr();
    assert(data != nullptr);
    D1Iter(i, 0U, rows() * cols()) data[i] += val;
    return *this;
  }
  Mtx& sub(T val){
    T* data = dataptr();
    assert(data != nullptr);
    D1Iter(i, 0U, rows() * cols()) data[i] -= val;
    return *this;
  }
  Mtx& mul(T val){
    T* data = dataptr();
    assert(data != nullptr);
    D1Iter(i, 0U, rows() * cols()) data[i] *= val;
    return *this;
  }
  Mtx& div(T val){
    T* data = dataptr();
    assert(data != nullptr);
    assert(val != T());
    D1Iter(i, 0U, rows() * cols()) data[i] /= val;
    return *this;
  }
  Mtx& pow(T val){
    T* data = dataptr();
    assert(data != nullptr);
    D1Iter(i, 0U, rows() * cols()) data[i] = std::pow(data[i], val);
    return *this;
  }
  template <typename F>
  Mtx& foreach(F&& transform){
    T* data = dataptr();
    assert(data != nullptr);
    D1Iter(i, 0U, rows() * cols()) transform(data[i]);
    return *this;
  }

  /* matrix operations */
  Mtx& add(const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);
    T* data = dataptr();
    T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) data[i] += odata[i];
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[index(ir, ic)] += odata[rindex(ir, ic)];
    else
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[rindex(ir, ic)] += odata[index(ir, ic)];
    return *this;
  }
  Mtx& sub(const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);
    T* data = dataptr();
    T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) data[i] -= odata[i];
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[index(ir, ic)] -= odata[rindex(ir, ic)];
    else
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[rindex(ir, ic)] -= odata[index(ir, ic)];
    return *this;
  }
  Mtx& mul(const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);
    T* data = dataptr();
    T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) data[i] *= odata[i];
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[index(ir, ic)] *= odata[rindex(ir, ic)];
    else
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[rindex(ir, ic)] *= odata[index(ir, ic)];
    return *this;
  }
  Mtx& div(const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);
    T* data = dataptr();
    T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) data[i] /= odata[i];
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[index(ir, ic)] /= odata[rindex(ir, ic)];
    else
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[rindex(ir, ic)] /= odata[index(ir, ic)];
    return *this;
  }
  Mtx& pow(const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);
    T* data = dataptr();
    T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) data[i] = std::pow(data[i], odata[i]);
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[index(ir, ic)] = std::pow(data[index(ir, ic)], odata[rindex(ir, ic)]);
    else
      D2IterC(ir, ic, 0, mRows, 0, mCols) data[rindex(ir, ic)] = std::pow(data[rindex(ir, ic)], odata[index(ir, ic)]);
    return *this;
  }
  template <typename F>
  Mtx& foreach(F transform, const Mtx& o){
    assert(mRows == o.mRows && mCols == o.mCols);

    T* data = dataptr();
    const T* odata = o.dataptr();
    if (is_rotated() == o.is_rotated())
      D1Iter(i, 0U, mRows * mCols) transform(data[i], odata[i]);
    else if (o.is_rotated() && not is_rotated())
      D2IterC(ir, ic, 0U, mRows, 0U, mCols) transform(data[index(ir, ic)], odata[rindex(ir, ic)]);
    else
      D2IterC(ir, ic, 0U, mRows, 0U, mCols) transform(data[rindex(ir, ic)], odata[index(ir, ic)]);
    return *this;
  }

  /* matrix multiplication */
  Mtx dot(const Mtx& o) const {
    assert(cols() == o.rows());

    Mtx ret(rows(), o.cols());
    T* a = dataptr();
    T* b = o.dataptr();
    T* c = ret.dataptr();

    o.unflip();

    if (is_rotated()){
      for (size_t i = 0; i < rows(); ++i)
        for (size_t j = 0; j < o.cols(); ++j)
          for (size_t k = 0; k < cols(); ++k)
            c[ret.rindex(i, j)] += a[rindex(i, k)] * b[o.index(k, j)];
      ret.flip_bit();
    } else {
      for (size_t i = 0; i < rows(); ++i)
        for (size_t j = 0; j < o.cols(); ++j)
          for (size_t k = 0; k < cols(); ++k)
            c[ret.index(i, j)] += a[index(i, k)] * b[o.index(k, j)];
    }
    return ret;
  }

  /* transpose functions */
  //modify self to become transpose
  Mtx& t(){
    if (is_rotated()) flip_bit();
    else              rotate();

    size_t tmp = mRows;
    mRows = mCols;
    mCols = tmp;

    return *this;
  }
  //return a copy of self that is transposed
  Mtx transpose(){
    Mtx ret = *this;
    if (ret.is_rotated()) ret.flip_bit();
    else                  ret.rotate();

    ret.mRows = mCols;
    ret.mCols = mRows;

    return ret;
  }

  /* return matrix to column matrix, undo internal matrix rotation */
  const Mtx& unflip() const {
    if (is_rotated()){
      rotate();
      flip_bit();
    }
    return *this;
  }

  /* change to internal rotation form */
  const Mtx& flip() const {
    if (not is_rotated()){
      rotate();
      flip_bit();
    }
    return *this;
  }

  /* statistics functions */
  vector<size_t> maxi(MtxDim dim) const {
    size_t n = dim == MCol ? mCols : mRows;
    vector<size_t> ret(n, 0);
    IndexFP indexfp = is_rotated() ? &Mtx::rindex : &Mtx::index;
    T* data = dataptr();

    if (dim == MCol){
      D2IterC(ir, ic, 0, rows(), 0, cols())
        if (data[(this->*indexfp)(ir, ic)] > data[(this->*indexfp)(ret[ic], ic)])
          ret[ic] = ir;
    } else {
      D2IterC(ir, ic, 0, rows(), 0, cols())
        if (data[(this->*indexfp)(ir, ic)] > data[(this->*indexfp)(ir, ret[ir])])
          ret[ir] = ic;
    }
    return ret;
  }
  vector<size_t> mini(MtxDim dim) const {
    size_t n = dim == MCol ? mCols : mRows;
    vector<size_t> ret(n, 0);
    IndexFP indexfp = is_rotated() ? &Mtx::rindex : &Mtx::index;
    T* data = dataptr();

    if (dim == MCol){
      D2IterC(ir, ic, 0, rows(), 0, cols())
        if (data[(this->*indexfp)(ir, ic)] < data[(this->*indexfp)(ret[ic], ic)])
          ret[ic] = ir;
    } else {
      D2IterC(ir, ic, 0, rows(), 0, cols())
        if (data[(this->*indexfp)(ir, ic)] < data[(this->*indexfp)(ir, ret[ir])])
          ret[ir] = ic;
    }
    return ret;
  }
  vector<T> sum(MtxDim dim) const {
    size_t n = dim == MCol ? mCols : mRows;
    vector<T> ret(n, 0.);
    IndexFP findex = is_rotated() ? &Mtx::rindex : &Mtx::index;
    T* data = dataptr();

    if (dim == MCol){
      D2IterC(ir, ic, 0, rows(), 0, cols()) ret[ic] += data[(this->*findex)(ir, ic)];
    } else {
      D2IterR(ir, ic, 0, rows(), 0, cols()) ret[ir] += data[(this->*findex)(ir, ic)];
    }
    return ret;
  }
  vector<T> mean(MtxDim dim) const {
    vector<T> ret = sum(dim);
    size_t n = dim == MCol ? mRows : mCols;
    D1Iter(i, 0, ret.size()) ret[i] /= (double)n;
    return ret;
  }

  /* save matrix to file */
  void save(const string& filename){
    ofstream out; out.open(filename.c_str());

    if (not out.is_open()){
      cout << "could not open file " << filename << " to save" << endl;
      return;
    }

    out << rows() << " " << cols() << " " << endl;

    T* data = dataptr();
    IndexFP indexp = is_rotated() ? &Mtx::rindex : &Mtx::index;

    for (size_t ic = 0; ic < cols(); ++ic){
      for (size_t ir = 0; ir < rows(); ++ir)
        out << data[(this->*indexp)(ir, ic)] << " ";
      out << endl;
    }

    out.close();
  }
};

template <typename T>
class RowRef {
  Mtx<T>& mMtx;
  size_t mRowId;
public:
  RowRef(Mtx<T>& m, size_t r): mMtx(m), mRowId(r) {}

  size_t size() const { return mMtx.cols(); }

  T operator[](size_t c) const {
    assert(c < mMtx.cols() && mRowId < mMtx.rows());
    if (not mMtx.is_rotated()) return mMtx.dataptr()[mMtx.index(mRowId, c)];
    else                       return mMtx.dataptr()[mMtx.rindex(mRowId, c)];
  }
  T& operator[](size_t c){
    assert(c < mMtx.cols() && mRowId < mMtx.rows());
    if (not mMtx.is_rotated()) return mMtx.dataptr()[mMtx.index(mRowId, c)];
    else                       return mMtx.dataptr()[mMtx.rindex(mRowId, c)];
  }

  RowRef& add(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.index(mRowId, i)] += v;
    else                       D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.rindex(mRowId, i)] += v;
    return *this;
  }
  RowRef& sub(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.index(mRowId, i)] -= v;
    else                       D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.rindex(mRowId, i)] -= v;
    return *this;
  }
  RowRef& mul(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.index(mRowId, i)] *= v;
    else                       D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.rindex(mRowId, i)] *= v;
    return *this;
  }
  RowRef& div(T v){
    assert(v != 0);
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.index(mRowId, i)] /= v;
    else                       D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.rindex(mRowId, i)] /= v;
    return *this;
  }
  RowRef& pow(T v){
    if (not mMtx.is_rotated())
      D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.index(mRowId, i)] = pow(mMtx.dataptr()[mMtx.index(mRowId, i)], v);
    else
      D1Iter(i, 0, mMtx.cols()) mMtx.dataptr()[mMtx.rindex(mRowId, i)] = pow(mMtx.dataptr()[mMtx.rindex(mRowId, i)], v);
    return *this;
  }
  template <typename F>
  RowRef& foreach(F transform){
    if (not mMtx.is_rotated())
      D1Iter(i, 0, mMtx.cols()) transform(mMtx.dataptr()[mMtx.index(mRowId, i)]);
    else
      D1Iter(i, 0, mMtx.cols()) transform(mMtx.dataptr()[mMtx.rindex(mRowId, i)]);
    return *this;
  }
};

template <typename T>
class ColRef {
  Mtx<T>& mMtx;
  size_t mColId;
public:
  ColRef(Mtx<T>& m, size_t c): mMtx(m), mColId(c) {}

  size_t size() const { return mMtx.rows(); }

  T operator[](size_t r) const {
    assert(r < mMtx.rows() && mColId < mMtx.cols());
    if (not mMtx.is_rotated()) return mMtx.dataptr()[mMtx.index(r, mColId)];
    else                       return mMtx.dataptr()[mMtx.rindex(r, mColId)];
  }
  T& operator[](size_t r){
    assert(r < mMtx.rows() && mColId < mMtx.cols());
    if (not mMtx.is_rotated()) return mMtx.dataptr()[mMtx.index(r, mColId)];
    else                       return mMtx.dataptr()[mMtx.rindex(r, mColId)];
  }

  ColRef& add(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.index(i, mColId)] += v;
    else                       D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.rindex(i, mColId)] += v;
    return *this;
  }
  ColRef& sub(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.index(i, mColId)] -= v;
    else                       D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.rindex(i, mColId)] -= v;
    return *this;
  }
  ColRef& mul(T v){
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.index(i, mColId)] *= v;
    else                       D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.rindex(i, mColId)] *= v;
    return *this;
  }
  ColRef& div(T v){
    assert(v != 0);
    if (not mMtx.is_rotated()) D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.index(i, mColId)] /= v;
    else                       D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.rindex(i, mColId)] /= v;
    return *this;
  }
  ColRef& pow(T v){
    if (not mMtx.is_rotated())
      D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.index(i, mColId)] = pow(mMtx.dataptr()[mMtx.index(i, mColId)], v);
    else
      D1Iter(i, 0, mMtx.rows()) mMtx.dataptr()[mMtx.rindex(i, mColId)] = pow(mMtx.dataptr()[mMtx.rindex(i, mColId)], v);
    return *this;
  }
  template <typename F>
  ColRef& foreach(F transform){
    if (not mMtx.is_rotated())
      D1Iter(i, 0, mMtx.rows()) transform(mMtx.dataptr()[mMtx.index(i, mColId)]);
    else
      D1Iter(i, 0, mMtx.rows()) transform(mMtx.dataptr()[mMtx.rindex(i, mColId)]);
    return *this;
  }
};

template <typename T>
const RowRef<T> Mtx<T>::row(size_t idx) const {
  return RowRef<T>(const_cast<Mtx<T>&>(*this), idx);
}
template <typename T>
RowRef<T> Mtx<T>::row(size_t idx){
  return RowRef<T>(*this, idx);
}
template <typename T>
const ColRef<T> Mtx<T>::col(size_t idx) const {
  return ColRef<T>(const_cast<Mtx<T>&>(*this), idx);
}
template <typename T>
ColRef<T> Mtx<T>::col(size_t idx){
  return ColRef<T>(*this, idx);
}

/* operators that does not generate new matrix instances */

/* operators that generate new matrix instances */
template <typename T>
Mtx<T> operator+(const Mtx<T>& a, const Mtx<T>& b){
  Mtx<T> ret(a);
  ret.add(b);
  return ret;
}
template <typename T>
Mtx<T> operator-(const Mtx<T>& a, const Mtx<T>& b){
  Mtx<T> ret(a);
  ret.sub(b);
  return ret;
}
template <typename T>
Mtx<T> operator*(const Mtx<T>& a, const Mtx<T>& b){
  return a.dot(b);
}
template <typename T>
Mtx<T> operator/(const Mtx<T>& a, const Mtx<T>& b){
  Mtx<T> ret(a);
  ret.div(b);
  return ret;
}

/* print matrix */
template <typename T>
ostream& operator<<(ostream& out, const Mtx<T>& m){
  for (size_t ir = 0; ir < m.rows(); ++ir){
    for (size_t ic = 0; ic < m.cols(); ++ic)
      out << m(ir, ic) << " ";
    out << endl;
  }
  return out;
}
template <typename T>
ostream& operator<<(ostream& out, const RowRef<T>& rr){
  D1Iter(i, 0, rr.size()) out << rr[i] << " ";
  out << endl;
  return out;
}
template <typename T>
ostream& operator<<(ostream& out, const ColRef<T>& cr){
  D1Iter(i, 0, cr.size()) out << cr[i] << " ";
  out << endl;
  return out;
}

#endif

#ifndef ML_EXPRTREE
#define ML_EXPRTREE

#include <cstring>
#include <cassert>
#include <sstream>

namespace ML {

class Mtx;

template <typename CRTP>
struct MtxBase {
  /* debug print */
  std::string print() const {
    std::stringstream ss;
    static_cast<const CRTP&>(*this).debug(ss);
    return ss.str();
  }
};

class MtxRef : public MtxBase<MtxRef> {
  const Mtx& mMtx;

public:
  MtxRef(const Mtx& m): MtxBase<MtxRef>(), mMtx(m) {}

  MtxRef(MtxRef&& o): MtxBase<MtxRef>(), mMtx(o.mMtx) {}
  MtxRef(const MtxRef&) = delete;
  ~MtxRef() = default;

  const Mtx& mtx() const {
    return mMtx;
  }

  void debug(std::stringstream& ss) const {
    ss << "Mtx";
  }
};

class Scl : public MtxBase<Scl> {
  double mW;

public:
  Scl(double d): MtxBase<Scl>(), mW(d) {}

  Scl(Scl&& o): MtxBase<Scl>(), mW(o.mW) {}
  Scl(const Scl&) = delete;
  ~Scl() = default;

  double val() const {
    return mW;
  }

  void debug(std::stringstream& ss) const {
    ss << "Scl";
  }
};

template <typename Op, typename X>
class Uop : public MtxBase<Uop<Op,X>> {
  X mx;

public:
  Uop(X&& x): MtxBase<Uop<Op,X>>(), mx(std::move(x)) {}
  Uop(Uop&& o): MtxBase<Uop<Op,X>>(), mx(std::move(o.mx)) {}
  Uop(const Uop&) = delete;
  ~Uop() = default;

  const X& param() const {
    return mx;
  }

  void debug(std::stringstream& ss) const {
    Op::debug(ss);
    mx.debug(ss);
  }
};

/* BEGIN Transpose */
struct TrnOp {
  static void debug(std::stringstream& ss){
    ss << "~";
  }
};

Uop<TrnOp, MtxRef> operator~(MtxRef&& a){
  return Uop<TrnOp, MtxRef>(std::move(a));
}

template <typename X>
Uop<TrnOp, X> operator~(MtxBase<X>&& a){
  return Uop<TrnOp, X>(static_cast<X&&>(a));
}
/* END Transpose */

template <typename Op, typename X, typename Y>
class Bop : public MtxBase<Bop<Op,X,Y>> {
  X mx;
  Y my;

public:
  Bop(X&& x, Y&& y): MtxBase<Bop<Op,X,Y>>(), mx(std::move(x)), my(std::move(y)) {}
  Bop(Bop&& o): MtxBase<Bop<Op, X, Y>>(), mx(std::move(o.mx)), my(std::move(o.my)) {}
  Bop(const Bop&) = delete;
  ~Bop() = default;

  const X& param1() const {
    return mx;
  }
  const Y& param2() const {
    return my;
  }

  void debug(std::stringstream& ss) const {
    ss << "(";
    mx.debug(ss);
    Op::debug(ss);
    my.debug(ss);
    ss << ")";
  }
};

/* BEGIN Add */
struct AddOp {
  static void debug(std::stringstream& ss){
    ss << "+";
  }
};

Bop<AddOp, MtxRef, Scl> operator+(MtxRef&& a, Scl&& b){
  return Bop<AddOp, MtxRef, Scl>(std::move(a), std::move(b));
}

Bop<AddOp, Scl, MtxRef> operator+(Scl&& a, MtxRef&& b){
  return Bop<AddOp, Scl, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<AddOp, Scl, X> operator+(Scl&& a, MtxBase<X>&& b){
  return Bop<AddOp, Scl, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X>
Bop<AddOp, X, Scl> operator+(MtxBase<X>&& a, Scl&& b){
  return Bop<AddOp, X, Scl>(static_cast<X&&>(a), std::move(b));
}

Bop<AddOp, MtxRef, MtxRef> operator+(MtxRef&& a, MtxRef&& b){
  return Bop<AddOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<AddOp, X, MtxRef> operator+(MtxBase<X>&& a, MtxRef&& b){
  return Bop<AddOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}

template <typename X>
Bop<AddOp, MtxRef, X> operator+(MtxRef&& a, MtxBase<X>&& b){
  return Bop<AddOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X, typename Y>
Bop<AddOp, X, Y> operator+(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<AddOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Add */

/* BEGIN Sub */
struct SubOp {
  static void debug(std::stringstream& ss){
    ss << "-";
  }
};

Bop<SubOp, MtxRef, Scl> operator-(MtxRef&& a, Scl&& b){
  return Bop<SubOp, MtxRef, Scl>(std::move(a), std::move(b));
}

Bop<SubOp, Scl, MtxRef> operator-(Scl&& a, MtxRef&& b){
  return Bop<SubOp, Scl, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<SubOp, Scl, X> operator-(Scl&& a, MtxBase<X>&& b){
  return Bop<SubOp, Scl, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X>
Bop<SubOp, X, Scl> operator-(MtxBase<X>&& a, Scl&& b){
  return Bop<SubOp, X, Scl>(static_cast<X&&>(a), std::move(b));
}

Bop<SubOp, MtxRef, MtxRef> operator-(MtxRef&& a, MtxRef&& b){
  return Bop<SubOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<SubOp, X, MtxRef> operator-(MtxBase<X>&& a, MtxRef&& b){
  return Bop<SubOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}

template <typename X>
Bop<SubOp, MtxRef, X> operator-(MtxRef&& a, MtxBase<X>&& b){
  return Bop<SubOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X, typename Y>
Bop<SubOp, X, Y> operator-(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<SubOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Sub */

/* BEGIN Element Multiply */
struct MulOp {
  static void debug(std::stringstream& ss){
    ss << "*";
  }
};

Bop<MulOp, MtxRef, Scl> operator*(MtxRef&& a, Scl&& b){
  return Bop<MulOp, MtxRef, Scl>(std::move(a), std::move(b));
}

Bop<MulOp, Scl, MtxRef> operator*(Scl&& a, MtxRef&& b){
  return Bop<MulOp, Scl, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<MulOp, Scl, X> operator*(Scl&& a, MtxBase<X>&& b){
  return Bop<MulOp, Scl, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X>
Bop<MulOp, X, Scl> operator*(MtxBase<X>&& a, Scl&& b){
  return Bop<MulOp, X, Scl>(static_cast<X&&>(a), std::move(b));
}

Bop<MulOp, MtxRef, MtxRef> operator*(MtxRef&& a, MtxRef&& b){
  return Bop<MulOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<MulOp, X, MtxRef> operator*(MtxBase<X>&& a, MtxRef&& b){
  return Bop<MulOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}

template <typename X>
Bop<MulOp, MtxRef, X> operator*(MtxRef&& a, MtxBase<X>&& b){
  return Bop<MulOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X, typename Y>
Bop<MulOp, X, Y> operator*(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<MulOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Element Multiply */

/* BEGIN Element Divide */
struct DivOp {
  static void debug(std::stringstream& ss){
    ss << "/";
  }
};

Bop<DivOp, MtxRef, Scl> operator/(MtxRef&& a, Scl&& b){
  return Bop<DivOp, MtxRef, Scl>(std::move(a), std::move(b));
}

Bop<DivOp, Scl, MtxRef> operator/(Scl&& a, MtxRef&& b){
  return Bop<DivOp, Scl, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<DivOp, Scl, X> operator/(Scl&& a, MtxBase<X>&& b){
  return Bop<DivOp, Scl, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X>
Bop<DivOp, X, Scl> operator/(MtxBase<X>&& a, Scl&& b){
  return Bop<DivOp, X, Scl>(static_cast<X&&>(a), std::move(b));
}

Bop<DivOp, MtxRef, MtxRef> operator/(MtxRef&& a, MtxRef&& b){
  return Bop<DivOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<DivOp, X, MtxRef> operator/(MtxBase<X>&& a, MtxRef&& b){
  return Bop<DivOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}

template <typename X>
Bop<DivOp, MtxRef, X> operator/(MtxRef&& a, MtxBase<X>&& b){
  return Bop<DivOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X, typename Y>
Bop<DivOp, X, Y> operator/(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<DivOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Element Divide */

/* BEGIN Matrix Multiplication */
struct DotOp {
  static void debug(std::stringstream& ss){
    ss << "^";
  }
};

Bop<DotOp, MtxRef, Scl> operator^(MtxRef&&, Scl&&){
  assert(!!!"Matrix multiply cannot be applied to a matrix and a scalar. Please use element wise multiply"); //TODO: why not static_assert
}
Bop<DotOp, Scl, MtxRef> operator^(Scl&&, MtxRef&&){
  assert(!!!"Matrix multiply cannot be applied to a matrix and a scalar. Please use element wise multiply");
}
template <typename X>
Bop<DotOp, Scl, X> operator^(Scl&&, MtxBase<X>&&){
  assert(!!!"Matrix multiply cannot be applied to a matrix and a scalar. Please use element wise multiply");
}
template <typename X>
Bop<DotOp, X, Scl> operator^(MtxBase<X>&&, Scl&&){
  assert(!!!"Matrix multiply cannot be applied to a matrix and a scalar. Please use element wise multiply");
}

Bop<DotOp, MtxRef, MtxRef> operator^(MtxRef&& a, MtxRef&& b){
  return Bop<DotOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}

template <typename X>
Bop<DotOp, X, MtxRef> operator^(MtxBase<X>&& a, MtxRef&& b){
  return Bop<DotOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}

template <typename X>
Bop<DotOp, MtxRef, X> operator^(MtxRef&& a, MtxBase<X>&& b){
  return Bop<DotOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}

template <typename X, typename Y>
Bop<DotOp, X, Y> operator^(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<DotOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Matrix Multiplication */

//TODO: additional operations need to be added

} //ML

#endif//ML_EXPRTREE

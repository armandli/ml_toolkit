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

/* BEGIN not */
struct NotOp {
  static void debug(std::stringstream& ss){
    ss << "not";
  }
};

Uop<NotOp, MtxRef> operator not(MtxRef&& a){
  return Uop<NotOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<NotOp, X> operator not(MtxBase<X>&& a){
  return Uop<NotOp, X>(static_cast<X&&>(a));
}
/* END not */

/* BEGIN Tanh */
struct TanhOp {
  static void debug(std::stringstream& ss){
    ss << "tanh";
  }
};

Uop<TanhOp, MtxRef> tanh(MtxRef&& a){
  return Uop<TanhOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<TanhOp, X> tanh(MtxBase<X>&& a){
  return Uop<TanhOp, X>(static_cast<X&&>(a));
}
/* END Tanh */

/* BEGIN softmax */
struct SoftmaxOp {
  static void debug(std::stringstream& ss){
    ss << "softmax";
  }
};

Uop<SoftmaxOp, MtxRef> softmax(MtxRef&& a){
  return Uop<SoftmaxOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<SoftmaxOp, X> softmax(MtxBase<X>&& a){
  return Uop<SoftmaxOp, X>(static_cast<X&&>(a));
}
/* END softmax */

/* BEGIN exp */
struct ExpOp {
  static void debug(std::stringstream& ss){
    ss << "exp";
  }
};

Uop<ExpOp, MtxRef> exp(MtxRef&& a){
  return Uop<ExpOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<ExpOp, X> exp(MtxBase<X>&& a){
  return Uop<ExpOp, X>(static_cast<X&&>(a));
}
/* END exp */

/* BEGIN isnan */
struct IsnanOp {
  static void debug(std::stringstream& ss){
    ss << "isnan";
  }
};

Uop<IsnanOp, MtxRef> isnan(MtxRef&& a){
  return Uop<IsnanOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<IsnanOp, X> isnan(MtxBase<X>&& a){
  return Uop<IsnanOp, X>(static_cast<X&&>(a));
}
/* END isnan */

/* BEGIN isnan then 0 */
struct Isnan0Op {
  static void debug(std::stringstream& ss){
    ss << "isnan0";
  }
};

Uop<Isnan0Op, MtxRef> isnan0(MtxRef&& a){
  return Uop<Isnan0Op, MtxRef>(std::move(a));
}
template <typename X>
Uop<Isnan0Op, X> isnan0(MtxBase<X>&& a){
  return Uop<Isnan0Op, X>(static_cast<X&&>(a));
}
/* END isnan then 0 */

/* BEGIN sqrt */
struct SqrtOp {
  static void debug(std::stringstream& ss){
    ss << "sqrt";
  }
};

Uop<SqrtOp, MtxRef> sqrt(MtxRef&& a){
  return Uop<SqrtOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<SqrtOp, X> sqrt(MtxBase<X>&& a){
  return Uop<SqrtOp, X>(static_cast<X&&>(a));
}
/* END sqrt */

/* BEGIN sum all */
struct SumOp {
  static void debug(std::stringstream& ss){
    ss << "sum";
  }
};

Uop<SumOp, MtxRef> sum(MtxRef&& a){
  return Uop<SumOp, MtxRef>(std::move(a));
}
template <typename X>
Uop<SumOp, X> sum(MtxBase<X>&& a){
  return Uop<SumOp, X>(static_cast<X&&>(a));
}
/* END sum all */

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

/* BEGIN Greater Than */
struct GtOp {
  static void debug(std::stringstream& ss){
    ss << ">";
  }
};

Bop<GtOp, MtxRef, Scl> operator>(MtxRef&& a, Scl&& b){
  return Bop<GtOp, MtxRef, Scl>(std::move(a), std::move(b));
}
Bop<GtOp, Scl, MtxRef> operator>(Scl&& a, MtxRef&& b){
  return Bop<GtOp, Scl, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<GtOp, Scl, X> operator>(Scl&& a, MtxBase<X>&& b){
  return Bop<GtOp, Scl, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X>
Bop<GtOp, X, Scl> operator>(MtxBase<X>&& a, Scl&& b){
  return Bop<GtOp, X, Scl>(static_cast<X&&>(a), std::move(b));
}
Bop<GtOp, MtxRef, MtxRef> operator>(MtxRef&& a, MtxRef&& b){
  return Bop<GtOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<GtOp, X, MtxRef> operator>(MtxBase<X>&& a, MtxRef&& b){
  return Bop<GtOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<GtOp, MtxRef, X> operator>(MtxRef&& a, MtxBase<X>&& b){
  return Bop<GtOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X, typename Y>
Bop<GtOp, X, Y> operator>(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<GtOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Greater Than */

/* BEGIN Greater Than Or 0 */
struct Gt0Op {
  static void debug(std::stringstream& ss){
    ss << ">0";
  }
};

Bop<Gt0Op, MtxRef, MtxRef> gt0(MtxRef&&, MtxRef&&){
  assert(!!!"gt0 can only be applied to a matrix and a constant");
}
template <typename X>
Bop<Gt0Op, X, MtxRef> gt0(MtxBase<X>&&, MtxRef&&){
  assert(!!!"gt0 can only be applied to a matrix and a constant");
}
template <typename X>
Bop<Gt0Op, MtxRef, X> gt0(MtxRef&&, MtxBase<X>&&){
  assert(!!!"gt0 can only be applied to a matrix and a constant");
}
template <typename X, typename Y>
Bop<Gt0Op, X, Y> gt0(MtxBase<X>&&, MtxBase<Y>&&){
  assert(!!!"gt0 can only be applied to a matrix and a constant");
}

Bop<Gt0Op, MtxRef, Scl> gt0(MtxRef&& a, Scl&& b){
  return Bop<Gt0Op, MtxRef, Scl>(std::move(a), std::move(b));
}
Bop<Gt0Op, Scl, MtxRef> gt0(Scl&& a, MtxRef&& b){
  return Bop<Gt0Op, Scl, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<Gt0Op, X, Scl> gt0(MtxBase<X>&& a, Scl&& b){
  return Bop<Gt0Op, X, Scl>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<Gt0Op, Scl, X> gt0(Scl&& a, MtxBase<X>&& b){
  return Bop<Gt0Op, Scl, X>(std::move(a), static_cast<X&&>(b));
}
/* END Greater Than Or 0 */

/* BEGIN Matrix Multiplication */
struct DotOp {
  static void debug(std::stringstream& ss){
    ss << "^";
  }
};

Bop<DotOp, MtxRef, Scl> operator^(MtxRef&&, Scl&&){
  assert(!!!"Matrix multiply cannot be applied to a matrix and a scalar. Please use element wise multiply");
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

/* BEGIN Mask */
struct MaskOp {
  static void debug(std::stringstream& ss){
    ss << "mask";
  }
};

Bop<MaskOp, MtxRef, Scl> mask(MtxRef&&, Scl&&){
  assert(!!!"mask cannot be applied to a matrix and a scalar");
}
Bop<MaskOp, Scl, MtxRef> mask(Scl&&, MtxRef&&){
  assert(!!!"mask cannot be applied to a matrix and a scalar");
}
template <typename X>
Bop<MaskOp, Scl, X> mask(Scl&&, MtxBase<X>&&){
  assert(!!!"mask cannot be applied to a matrix and a scalar");
}
template <typename X>
Bop<MaskOp, X, Scl> mask(MtxBase<X>&&, Scl&&){
  assert(!!!"mask cannot be applied to a matrix and a scalar");
}

Bop<MaskOp, MtxRef, MtxRef> mask(MtxRef&& a, MtxRef&& b){
  return Bop<MaskOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<MaskOp, X, MtxRef> mask(MtxBase<X>&& a, MtxRef&& b){
  return Bop<MaskOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<MaskOp, MtxRef, X> mask(MtxRef&& a, MtxBase<X>&& b){
  return Bop<MaskOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X, typename Y>
Bop<MaskOp, X, Y> mask(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<MaskOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Mask */

/* BEGIN Derivaitive of RELU */
struct DReluOp {
  static void debug(std::stringstream& ss){
    ss << "drelu";
  }
};

Bop<DReluOp, MtxRef, Scl> drelu(MtxRef&&, Scl&&){
  assert(!!!"drelu cannot be applied to a matrix and a scalar");
}
Bop<DReluOp, Scl, MtxRef> drelu(Scl&&, MtxRef&&){
  assert(!!!"drelu cannot be applied to a matrix and a scalar");
}
template <typename X>
Bop<DReluOp, X, Scl> drelu(MtxBase<X>&&, Scl&&){
  assert(!!!"drelu cannot be applied to a matrix and a scalar");
}
template <typename X>
Bop<DReluOp, Scl, X> drelu(Scl&&, MtxBase<X>&&){
  assert(!!!"drelu cannot be applied to a matrix and a scalar");
}

Bop<DReluOp, MtxRef, MtxRef> drelu(MtxRef&& a, MtxRef&& b){
  return Bop<DReluOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<DReluOp, X, MtxRef> drelu(MtxBase<X>&& a, MtxRef&& b){
  return Bop<DReluOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<DReluOp, MtxRef, X> drelu(MtxRef&& a, MtxBase<X>&& b){
  return Bop<DReluOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X, typename Y>
Bop<DReluOp, X, Y> drelu(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<DReluOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Deriviative of RELU */

/* BEGIN Cross Entropy Loss */
struct CrossEntropyLossOp {
  static void debug(std::stringstream& ss){
    ss << "ce_loss";
  }
};

Bop<CrossEntropyLossOp, MtxRef, Scl> ce_loss(MtxRef&&, Scl&&){
  assert(!!!"cross entropy loss can only be applied to 2 matrcies");
}
Bop<CrossEntropyLossOp, Scl, MtxRef> ce_loss(Scl&&, MtxRef&&){
  assert(!!!"cross entropy loss can only be applied to 2 matrcies");
}
template <typename X>
Bop<CrossEntropyLossOp, X, Scl> ce_loss(MtxBase<X>&&, Scl&&){
  assert(!!!"cross entropy loss can only be applied to 2 matrcies");
}
template <typename X>
Bop<CrossEntropyLossOp, Scl, X> ce_loss(Scl&&, MtxBase<X>&&){
  assert(!!!"cross entropy loss can only be applied to 2 matrcies");
}

Bop<CrossEntropyLossOp, MtxRef, MtxRef> ce_loss(MtxRef&& a, MtxRef&& b){
  return Bop<CrossEntropyLossOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<CrossEntropyLossOp, X, MtxRef> ce_loss(MtxBase<X>&& a, MtxRef&& b){
  return Bop<CrossEntropyLossOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<CrossEntropyLossOp, MtxRef, X> ce_loss(MtxRef&& a, MtxBase<X>&& b){
  return Bop<CrossEntropyLossOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X, typename Y>
Bop<CrossEntropyLossOp, X, Y> ce_loss(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<CrossEntropyLossOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Cross Entropy Loss */

/* BEGIN Cross Entropy Accuracy */
struct CrossEntropyAccuracyOp {
  static void debug(std::stringstream& ss){
    ss << "ce_accuracy";
  }
};

Bop<CrossEntropyAccuracyOp, MtxRef, Scl> ce_accuracy(MtxRef&&, Scl&&){
  assert(!!!"cross entropy accuracy can only be applied to 2 matrices");
}
Bop<CrossEntropyAccuracyOp, Scl, MtxRef> ce_accuracy(Scl&&, MtxRef&&){
  assert(!!!"cross entropy accuracy can only be applied to 2 matrices");
}
template <typename X>
Bop<CrossEntropyAccuracyOp, X, Scl> ce_accuracy(MtxBase<X>&&, Scl&&){
  assert(!!!"cross entropy accuracy can only be applied to 2 matrices");
}
template <typename X>
Bop<CrossEntropyAccuracyOp, Scl, X> ce_accuracy(Scl&&, MtxBase<X>&&){
  assert(!!!"cross entropy accuracy can only be applied to 2 matrices");
}

Bop<CrossEntropyAccuracyOp, MtxRef, MtxRef> ce_accuracy(MtxRef&& a, MtxRef&& b){
  return Bop<CrossEntropyAccuracyOp, MtxRef, MtxRef>(std::move(a), std::move(b));
}
template <typename X>
Bop<CrossEntropyAccuracyOp, X, MtxRef> ce_accuracy(MtxBase<X>&& a, MtxRef&& b){
  return Bop<CrossEntropyAccuracyOp, X, MtxRef>(static_cast<X&&>(a), std::move(b));
}
template <typename X>
Bop<CrossEntropyAccuracyOp, MtxRef, X> ce_accuracy(MtxRef&& a, MtxBase<X>&& b){
  return Bop<CrossEntropyAccuracyOp, MtxRef, X>(std::move(a), static_cast<X&&>(b));
}
template <typename X, typename Y>
Bop<CrossEntropyAccuracyOp, X, Y> ce_accuracy(MtxBase<X>&& a, MtxBase<Y>&& b){
  return Bop<CrossEntropyAccuracyOp, X, Y>(static_cast<X&&>(a), static_cast<Y&&>(b));
}
/* END Cross Entropy Accuracy */

//TODO: expand operation here

} //ML

#endif//ML_EXPRTREE

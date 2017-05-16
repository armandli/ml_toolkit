#ifndef ML_FUNCTIONS
#define ML_FUNCTIONS

#include <cassert>
#include <cmath>

#include <ml_common.h>
#include <ml_matrix.h>

namespace ML {

/* Cost Functions */
struct MSE {
  double loss(const Mtx& O, const Mtx& Y){
    double loss = O.binary_reduce([](double o, double y){
        double d = o - y;
        return d * d;
    }, Y);
    loss /= (double)O.rows();
    return loss;
  }
  double accuracy(const Mtx& O, const Mtx& Y){
    double acc = O.binary_reduce([](double o, double y){
        double d = o - y;
        return d * d;
    }, Y);
    return sqrt(acc) * 0.5 / (double)O.rows();
  }
  Mtx deriviative(const Mtx& O, const Mtx& Y){
    Mtx dY = O - Y;
    double sz = Y.rows();
    dY.unary_expr([&sz](double d){
        double ret = d / sz;
        if (std::isnan(ret)) ret = 0.;
        return ret;
    });
    return dY;
  }
  void classification(Mtx&){/*NOOP*/}
};

template <int F>
struct MSVM_ {
  double loss(const Mtx& O, const Mtx& Y){
    assert(O.rows() == Y.rows() && O.cols() == Y.cols());

    std::vector<DimV> mcoeffs = Y.max_coeff(MRow);
    //TODO: this double loop is pretty bad
    for (size_t ir = 0; ir < O.rows(); ++ir){
      double v = O(ir, mcoeffs[ir].idx);
      double sum = 0.;
      for (size_t ic = 0; ic < O.cols(); ++ic){
        double e = O(ir, ic) - v + (double)F;
        if (e < 0.) e = 0.;
        sum += e * e;
      }
      loss += sum - (double)F * (double)F;
    }
    loss /= (double)Y.rows();
    return loss;
  }
  double accuracy(const Mtx& O, const Mtx& Y){
    assert(O.rows() == Y.rows() && O.cols() == Y.cols());

    std::vector<DimV> ocoeff = O.max_coeff(MRow);
    std::vector<DimV> ycoeff = Y.max_coeff(MRow);
    int count = 0;
    for (size_t i = 0; i < O.rows(); ++i)
      if (ocoeff[i].idx == ycoeff[i].idx)
        count++;
    return (double)count / (double)Y.rows();
  }
  Mtx deriviative(const Mtx& O, const Mtx& Y){
    Mtx dY = O - Y;
    double sz = Y.rows();
    dY.unary_expr([&sz](double d){
        double ret = d / sz;
        if (std::isnan(ret)) ret = 0.;
        return ret;
    });
    return dY;
  }
  void classification(Mtx&){/*NOOP*/}
};
using MSVM = MSVM_<1>;

void softmax(Mtx& m, MtxDim dim){
  m = m.unary_expr([](double d){ return exp(d); }); 
  switch (dim){
  case MRow: {
    Mtx rowsum = m.reduce([](const SubMtx& sm){
        double sum = 0.;
        for (size_t i = 0; i < sm.cols(); ++i)
          sum += sm(0, i);
        return sum;
    }, MRow);
    //TODO: anything else I can think of for matrix class to have so this is not a loop?
    for (size_t i = 0; i < m.rows(); ++i){
      //TODO: maybe SubMtx should accept rvalue reference operator as well
      SubMtx sm = m.row(i);
      sm /= rowsum(i, 0);
    }
  } break; //MRow
  case MCol: {
    Mtx colsum = m.reduce([](const SubMtx& sm){
        double sum = 0.;
        for (size_t i = 0; i < sm.rows(); ++i)
          sum += sm(i, 0);
        return sum;
    }, MCol);
    for (size_t i = 0; i < m.cols(); ++i){
      SubMtx sm = m.col(i);
      sm /= colsum(0, i);
    }
  } break; //MCol
  default: assert(false);
  } //switch
}

struct CrossEntropy {
  double loss(const Mtx& O, const Mtx& Y){ 
    assert(O.rows() == Y.rows() && O.cols() == Y.cols());

    double loss = 0.;
    std::vector<DimV> ycoeffs = Y.max_coeff(MRow);
    for (size_t i = 0; i < O.rows(); ++i)
      loss += -1. * log(O(i, ycoeffs[i].idx));
    loss /= (double)Y.rows();
    return loss;
  }
  double accuracy(const Mtx& O, const Mtx& Y){ 
    assert(O.rows() == Y.rows() && O.cols() == Y.cols());

    std::vector<DimV> ycoeffs = Y.max_coeff(MRow);
    std::vector<DimV> ocoeffs = O.max_coeff(MRow);
    size_t count = 0;
    for (size_t i = 0; i < O.rows(); ++i)
      if (ycoeffs[i].idx == ocoeffs[i].idx)
        count++;
    return (double)count / (double)Y.rows();
  }
  Mtx deriviative(const Mtx& O, const Mtx& Y){
    Mtx dY = O - Y;
    double sz = Y.rows();
    dY.unary_expr([&sz](double d){
        double ret = d / sz;
        if (std::isnan(ret)) ret = 0.;
        return ret;
    });
    return dY;
  }
  void classification(Mtx& O){
    softmax(O, MRow);
  }
};

/* Activation Functions */
struct LinearFun {
  void function(Mtx&){ /*do nothing*/ }
  void deriviative(Mtx&, Mtx&){ /*do nothing*/ }
};

struct SigmoidFun {
  void function(Mtx& m){
    m.unary_expr([](double d){ return 1. / (1. + exp(d * -1.)); });
  }
  void deriviative(Mtx& dm, Mtx& m){
    dm.binary_expr([](double a, double b){ return a * b * (1. - b); }, m);
  }
};

struct TanhFun {
  void function(Mtx& m){
    m.unary_expr([](double d){ return tanh(d); });
  }
  void deriviative(Mtx& dm, Mtx& m){
    dm.binary_expr([](double a, double b){ return a * (1. - b * b); }, m);
  }
};

struct ReluFun {
  void function(Mtx& m){
    m.unary_expr([](double d){ return d > 0. ? d : 0.; });
  }
  void deriviative(Mtx& dm, Mtx& m){
    dm.binary_expr([](double a, double b){ return b > 0. ? a : 0.; }, m);
  }
};

/* Regularization Functions */
template <long MNom, long MDnom>
class MaxNormReg_ {
  double maxval;
public:
  explicit MaxNormReg_(double maxval = (double)MNom / (double)MDnom): maxval(maxval) {}
  double loss(Mtx&, double){
    return 0.; 
  }
  void regularize(Mtx& dW, Mtx&, double){
    dW.unary_expr([this](double d){ 
        if (d > maxval) return maxval;
        if (d < -1. * maxval) return maxval * -1.;
        return d;
    }); 
  }
};
using MaxNormReg = MaxNormReg_<5, 1>; 

struct L2Reg {
  double loss(Mtx& W, double reg){
    Mtx D(W);
    D.unary_expr([](double d){
      return d * d;
    });
    double sum = D.sum();
    return sum * 0.5 * reg;
  }
  void regularize(Mtx& dW, Mtx& W, double reg){
    dW.binary_expr([&reg](double d, double w){ 
        return d + reg * w;
    }, W); 
  }
};

/* dropout */
Mtx dropout_mask(int r, int c, double p = 0.5){
  Mtx mask = Mtx::random<std::uniform_real_distribution<double>>(r, c, std::uniform_real_distribution<double>(0., 1.));
  mask.unary_expr([&p](double d){
      if (d < p) return 0.;
      else       return 1. / p;
  });
  return mask;
}

template <long PNom, long PDnom>
void apply_dropout_(Mtx& H){
  Mtx U = dropout_mask(H.rows(), H.cols(), (double)PNom / (double)PDnom);
  H.binary_expr([](double h, double u){
      return h * u;
  }, U);
}
void apply_dropout(Mtx& H){
  apply_dropout_<1, 2>(H);
}

/* Parameter Update Functions */
struct SimpleUpdate {
  SimpleUpdate(int, int){}
  void update(Mtx& W, const Mtx& dW, double lrate){
    W.binary_expr([lrate](double w, double d){
        return w + -1. * lrate * d;
    }, dW);
  }
};

template <long MUNom, long MUDnom>
class MomentumUpdate_ {
  Mtx V;
  double mu;
public:
  MomentumUpdate_(int rsize, int csize, double mu = (double)MUNom / (double)MUDnom): V(Mtx::zeros(rsize, csize)), mu(mu) {}
  void update(Mtx& W, const Mtx& dW, double lrate){
    V.binary_expr([this, lrate](double v, double w){
        return v * mu - w * lrate;
    }, dW);
    W += V;
  }
};
using MomentumUpdate = MomentumUpdate_<9, 10>;

template <long MUNom, long MUDnom>
class NesterovUpdate_ {
  Mtx Vp, V;
  double mu;
public:
  NesterovUpdate_(int rsize, int csize, double mu = (double)MUNom / (double)MUDnom):
    Vp(Mtx::zeros(rsize, csize)), V(Mtx::zeros(rsize, csize)), mu(mu) {}
  void update(Mtx& W, const Mtx& dW, double lrate){
    Vp = V;
    V.binary_expr([this, lrate](double v, double w){
        return v * mu - w * lrate;
    }, dW);
    W = W.ternary_expr([this](double w, double p, double v){
        return w + p * (mu * -1.) + v * (1. + mu);
    }, Vp, V);
  }
};
using NesterovUpdate = NesterovUpdate_<9, 10>;

class AdagradUpdate {
  Mtx mW;
public:
  AdagradUpdate(int rsize, int csize): mW(Mtx::zeros(rsize, csize)) {}
  void update(Mtx& W, const Mtx& dW, double lrate){
    mW.binary_expr([](double m, double d){
        return m + d * d;
    }, dW);
    W.ternary_expr([lrate](double w, double m, double d){
        return w + -1. * lrate * d / (sqrt(m) + 1e-8);
    }, mW, dW);
  }
};

template <long Dnom, long DDnom>
class RMSPropUpdate_ {
  Mtx mW;
  double decay;
public:
  RMSPropUpdate_(int rsize, int csize, double decay = (double)Dnom / (double)DDnom): mW(Mtx::zeros(rsize, csize)), decay(decay) {}
  void update(Mtx& W, const Mtx& dW, double lrate){
    mW.binary_expr([this](double m, double d){
        return decay * m + (1. - decay) * (d * d);
    }, dW);
    W.ternary_expr([lrate](double w, double m, double d){
        return w + -1. * lrate * d / (sqrt(m) + 1e-8);
    }, mW, dW);
  }
};
using RMSPropUpdate = RMSPropUpdate_<99, 100>;

template <long B1Nom, long B2Nom, long Dnom>
class AdamUpdate_ {
  Mtx M, V;
  double b1, b2;
public:
  AdamUpdate_(int rsize, int csize, double b1 = (double)B1Nom / (double)Dnom, double b2 = (double)B2Nom / (double)Dnom):
    M(Mtx::zeros(rsize, csize)), V(Mtx::zeros(rsize, csize)), b1(b1), b2(b2) {}
  void update(Mtx& W, const Mtx& dW, double lrate){
    M.binary_expr([this](double m, double d){
        return b1 * m + (1. - b1) * d;
    }, dW);
    V.binary_expr([this](double v, double d){
        return b2 * v + (1. - b2) * (d * d);
    }, dW);
    W.ternary_expr([lrate](double w, double m, double v){
        return w + -1. * lrate * m / (sqrt(v) + 1e-8);
    }, M, V);
  }
};
using AdamUpdate = AdamUpdate_<9000, 9990, 10000>;

/* Learning Rate Functions */
//TODO

} //ML

#endif //ML_FUNCTIONS

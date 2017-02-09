#ifndef FFN_UTIL
#define FFN_UTIL

#include <cmath>
#include <ctime>
#include <random>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

enum MatrixDim {
  RowDim,
  ColDim,
};

MatrixXd random_matrix(int rows, int cols){
  normal_distribution<double> dist(0.0, 0.5);
  default_random_engine eng(time(0));
  MatrixXd ret(rows, cols);

  ret = ret.unaryExpr([&dist, &eng](double){ return dist(eng); });
  return ret;
}

/* Activation Functions */
struct LinearFun {
  void function(MatrixXd&){ /* do nothing */ }
  void deriviative(MatrixXd&, MatrixXd&){ /* do nothing */ }
};

struct SigmoidFun {
  void function(MatrixXd& m){
    m = m.unaryExpr([](double d){ return 1. / (1. + exp(d * -1.)); });
  }
  void deriviative(MatrixXd& dm, MatrixXd& m){
    dm = dm.binaryExpr(m, [](double a, double b){ return a * b * (1. - b); });
  }
};

struct TanhFun {
  void function(MatrixXd& m){
    m = m.unaryExpr([](double d){ return tanh(d); });
  }
  void deriviative(MatrixXd& dm, MatrixXd& m){
    dm = dm.binaryExpr(m, [](double a, double b){ return a * (1. - b * b); });
  }
};

struct ReluFun {
  void function(MatrixXd& m){
    m = m.unaryExpr([](double d){ return d > 0. ? d : 0.; });
  }
  void deriviative(MatrixXd& dm, MatrixXd& m){
    dm = dm.binaryExpr(m, [](double a, double b){ return b > 0. ? a : 0.; });
  }
};

/* Regularization Functions */
template <long MNom = 5, long MDnom = 1>
class MaxNormReg {
  double maxval;
public:
  explicit MaxNormReg(double maxval = (double)MNom / (double)MDnom): maxval(maxval) {}
  double loss(MatrixXd&, double){
    return 0.;
  }
  void regularize(MatrixXd& dW, MatrixXd&, double){
    dW = dW.unaryExpr([this](double d){
        if (d > maxval) return maxval;
        if (d < -1. * maxval) return maxval * -1.;
        return d;
    });
  }
};

struct L2Reg {
  double loss(MatrixXd& W, double reg){
    MatrixXd D = W.unaryExpr([](double d){
        return d * d;
    });
    double loss = D.array().sum();
    return loss * 0.5 * reg;
  }
  void regularize(MatrixXd& dW, MatrixXd& W, double reg){
    dW = dW.binaryExpr(W, [&reg](double d, double w){
        return d + reg * w;
    });
  }
};

/* dropout */
MatrixXd dropout_mask(int r, int c, double p = 0.5){
  MatrixXd mask = MatrixXd::Random(r, c);
  mask = mask.unaryExpr([&p](double d){
      d = abs(d);
      if (d < p) return 0.;
      else       return 1. / p;
  });
  return mask;
}

template <long PNom = 1, long PDnom = 2>
void apply_dropout(MatrixXd& H){
  MatrixXd U = dropout_mask(H.rows(), H.cols(), (double)PNom / (double)PDnom);
  H = H.binaryExpr(U, [](double h, double u){
      return h * u;
  });
}

/* Parameter Update Functions */
struct SimpleUpdate {
  SimpleUpdate(int, int){}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    W = W.binaryExpr(dW, [&lrate](double w, double d){
        return w + -1. * lrate * d;
    });
  }
};

template <long MUNom = 9, long MUDnom = 10>
class MomentumUpdate {
  MatrixXd V;
  double mu;
public:
  MomentumUpdate(int rsize, int csize, double mu = (double)MUNom / (double)MUDnom): V(MatrixXd::Zero(rsize, csize)), mu(mu) {}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    V = V * mu - dW * lrate;
    W += V;
  }
};

template <long MUNom = 9, long MUDnom = 10>
class NesterovUpdate {
  MatrixXd Vp, V;
  double mu;
public:
  NesterovUpdate(int rsize, int csize, double mu = (double)MUNom / (double)MUDnom):
    Vp(MatrixXd::Zero(rsize, csize)), V(MatrixXd::Zero(rsize, csize)), mu(mu) {}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    Vp = V;
    V = V * mu - dW * lrate;
    W += Vp * (mu * -1.) + V * (1. + mu);
  }
};

class AdagradUpdate {
  MatrixXd mW;
public:
  AdagradUpdate(int rsize, int csize): mW(MatrixXd::Zero(rsize, csize)) {}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    mW = mW.binaryExpr(dW, [](double m, double d){
        return m + d * d;
    });
    MatrixXd smW = mW.binaryExpr(dW, [&lrate](double m, double d){
        return -1. * lrate * d / (sqrt(m) + 1e-8);
    });
    W += smW;
  }
};

template <long Dnom = 99, long DDnom = 100>
class RMSPropUpdate {
  MatrixXd mW;
  double decay;
public:
  RMSPropUpdate(int rsize, int csize, double decay = (double)Dnom / (double)DDnom): mW(MatrixXd::Zero(rsize, csize)), decay(decay) {}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    mW = mW.binaryExpr(dW, [this](double m, double d){
        return decay * m + (1. - decay) * (d * d);
    });
    MatrixXd smW = mW.binaryExpr(dW, [&lrate](double m, double d){
        return -1. * lrate * d / (sqrt(m) + 1e-8);
    });
    W += smW;
  }
};

template <long B1Nom = 9000, long B2Nom = 9990, long Dnom = 10000>
class AdamUpdate {
  MatrixXd M, V;
  double b1, b2;
public:
  AdamUpdate(int rsize, int csize, double b1 = (double)B1Nom / (double)Dnom, double b2 = (double)B2Nom / (double)Dnom):
    M(MatrixXd::Zero(rsize, csize)), V(MatrixXd::Zero(rsize, csize)), b1(b1), b2(b2) {}
  void update(MatrixXd& W, const MatrixXd& dW, double lrate){
    M = M.binaryExpr(dW, [this](double m, double d){
        return b1 * m + (1. - b1) * d;
    });
    V = V.binaryExpr(dW, [this](double v, double d){
        return b2 * v + (1. - b2) * (d * d);
    });
    MatrixXd S = M.binaryExpr(V, [&lrate](double m, double v){
        return -1. * lrate * m / (sqrt(v) + 1e-8);
    });
    W += S;
  }
};

//TODO: learning rate decay functions
/* Learning Rate Functions */
class ConstantRate {
  double learn;
public:
  ConstantRate(double init, int): learn(init){}
  double rate(){ return learn; }
};

class LinearRate {
  double learn;
  int iterations;
public:
  LinearRate(double init, int iterations): learn(init), iterations(iterations) {}
  double rate(){ return learn; } //TODO
};

/* Cost Functions */
struct MSE {
  double loss(const MatrixXd& O, const MatrixXd& Y){
    MatrixXd D = O - Y;
    D = D.unaryExpr([](double d){ return d * d; });
    double loss = D.array().sum() / (double)O.rows();
    return loss;
  }
  double accuracy(const MatrixXd& O, const MatrixXd& Y){
    MatrixXd D = O - Y;
    D = D.unaryExpr([](double d){ return d * d; });
    return sqrt(D.array().sum()) * 0.5 / (double)O.rows();
  }
  MatrixXd deriviative(const MatrixXd& O, const MatrixXd& Y){
    MatrixXd dY = O - Y;
    double sz = Y.rows();
    dY = dY.unaryExpr([&sz](double d){ return d / sz; });
    return dY;
  }
  void classification(MatrixXd&){/*NOOP*/}
};

template <int F = 1>
struct MSVM {
  double loss(const MatrixXd& O, const MatrixXd& Y){
    double loss = 0.;
    for (int i = 0; i < O.rows(); ++i){
      MatrixXd::Index midx; Y.row(i).maxCoeff(&midx);
      double v = O(i, midx);
      MatrixXd D = O.row(i).unaryExpr([&v](double d){
          double e = d - v + (double)F;
          if (e < 0.) e = 0.;
          return e * e;
      });
      loss += D.array().sum() - (double)F * (double)F;
    }
    loss /= (double)Y.rows();
    return loss;
  } 
  double accuracy(const MatrixXd& O, const MatrixXd& Y){
    int count = 0;
    for (int i = 0; i < O.rows(); ++i){
      MatrixXd::Index idxO, idxY;
      Y.row(i).maxCoeff(&idxY);
      O.row(i).maxCoeff(&idxO);
      if (idxO == idxY) count++;
    }
    return (double)count / (double)Y.rows();
  }
  MatrixXd deriviative(const MatrixXd& O, const MatrixXd& Y){
    MatrixXd dY = O - Y;
    double sz = Y.rows();
    dY = dY.unaryExpr([&sz](double d){ return d / sz; });
    return dY;
  }
  void classification(MatrixXd&){/*NOOP*/}
};

void softmax(MatrixXd& m, MatrixDim dim){
  m = m.unaryExpr([](double d){ return exp(d); });
  if (dim == RowDim){
    MatrixXd rowsum = m.rowwise().sum();
    for (int i = 0; i < m.rows(); ++i)
      m.row(i) /= rowsum(i, 0);
  } else {
    MatrixXd colsum = m.colwise().sum();
    for (int i = 0; i < m.cols(); ++i)
      m.col(i) /= colsum(0, i);
  }
}

struct CrossEntropy {
  double loss(const MatrixXd& O, const MatrixXd& Y){
    double loss = 0.;
    for (int i = 0; i < O.rows(); ++i){
      MatrixXd::Index maxIdx;
      Y.row(i).maxCoeff(&maxIdx);
      loss += -1. * log(O(i, maxIdx));
    }
    loss /= (double)Y.rows();
    return loss;
  }
  double accuracy(const MatrixXd& O, const MatrixXd& Y){
    int count = 0;
    for (int i = 0; i < O.rows(); ++i){
      MatrixXd::Index idxO, idxY;
      Y.row(i).maxCoeff(&idxY);
      O.row(i).maxCoeff(&idxO);
      if (idxO == idxY) count++;
    }
    return (double)count / (double)Y.rows();
  }
  MatrixXd deriviative(const MatrixXd& O, const MatrixXd& Y){
    MatrixXd dY = O - Y;
    double sz = Y.rows();
    dY = dY.unaryExpr([&sz](double d){ return d / sz; });
    return dY;
  }
  void classification(MatrixXd& O){
    softmax(O, MatrixDim::RowDim);
  }
};

#endif

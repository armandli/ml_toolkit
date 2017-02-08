#ifndef ML_UTIL
#define ML_UTIL

#include <matrix.h>
#include <cmath>
#include <random>
using namespace std;

Mtx<double> random_matrix(size_t r, size_t c){
  Mtx<double> ret(r, c, 0.);
  normal_distribution<double> dist(0., 0.5);
  default_random_engine engine;

  ret.foreach([&](double& d){
      d = dist(engine);
  });
  return ret;
}

/* activation functions */
struct SigmoidFun {
  static double sigmoid(double v){
    return 1. / (1. + exp(-1. * v));
  }
  void function(Mtx<double>& m){
    m.foreach([](double& d){
        d = sigmoid(d);
    });
  }
  void deriviative(Mtx<double>& dm, const Mtx<double>& m){
    dm.foreach([](double& d, const double& k){
        d = k * (1. - k);
    }, m);
  }
};

struct TanhFun {
  void function(Mtx<double>& m){
    m.foreach([](double& d){
        d = tanh(d);
    });
  }
  void deriviative(Mtx<double>& dm, const Mtx<double>& m){
    dm.foreach([](double& d, const double& k){
        d = 1 - k * k;
    }, m);
  }
};

struct ReluFun {
  void function(Mtx<double>& m){
    m.foreach([](double& d){
        if (d <= 0.) d = 0.;
    });
  }
  void deriviative(Mtx<double>& dm, const Mtx<double>& m){
    dm.foreach([](double& d, const double& k){
        d = k <= 0. ? 0. : d;
    }, m);
  }
};

/* softmax output */
void softmax(Mtx<double>& m, MtxDim dim){
  m.foreach([](double& d){
      d = exp(d);
  });
  vector<double> sums = m.sum(dim);
  
  if (dim == MCol){
    D1Iter(i, 0, sums.size()) {
      ColRef<double> c = m.col(i);
      c.foreach([&sums, i](double& d){
          d = d / sums[i];
      });
    }
  } else {
    D1Iter(i, 0, sums.size()) {
      RowRef<double> r = m.row(i);
      r.foreach([&sums, i](double& d){
          d = d / sums[i];
      });
    }
  }
}

/* base class for machine learning */
template <typename ACT>
struct SNN {
  Mtx<double> Wxh;
  Mtx<double> Who;
  Mtx<double> H;   //cached hidden state

  SNN(size_t din, size_t dout, size_t dh){
    //training initialization
    Wxh = random_matrix(din, dh);
    Who = random_matrix(dh, dout);
  }

  Mtx<double> feed_forward(const Mtx<double>& x){
    ACT activation;

    H = x * Wxh;
    activation.function(H);
    Mtx<double> y = H * Who;
    softmax(y, MRow);
    return y;
  }

  void save(const string& mxh_filename, const string& mho_filename){
    Wxh.save(mxh_filename);
    Who.save(mho_filename);
  }
};

#endif

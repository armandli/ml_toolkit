#ifndef ML_FFN
#define ML_FFN

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <ml_matrix.h>
#include <ml_functions.h>

namespace ML {

template <typename Cost, typename Regularization, typename ParamUpdate>
class FFN1 {
  Mtx W;
  ParamUpdate updateW;
  Cost cost;
  Regularization regularization;
  double learning_rate;
  double regularization_rate;
  size_t iterations;
  bool display_loss;

  Mtx feed_forward(const Mtx& X){
    Mtx O = X * W;
    cost.classification(O);
    return O;
  }
  double compute_loss(const Mtx& O, const Mtx& Y, size_t icount){
    double data_loss, reg_loss;

    data_loss = cost.loss(O, Y);
    reg_loss = regularization.loss(W, regularization_rate);

    if (display_loss && icount % (iterations / 100) == 0)
      std::cout << "data: " << data_loss << " reg: " << reg_loss << std::endl;

    return data_loss + reg_loss;
  }

  void back_propagation(const Mtx& X, const Mtx& O, const Mtx& Y, Mtx& dW){
    Mtx dY = cost.deriviative(O, Y);
    dW = X.transpose() * dY;
    regularization.regularize(dW, W, regularization_rate);
  }

  void param_update(const Mtx& dW){
    updateW.update(W, dW, learning_rate);
  }
  double test_accuracy(const Mtx& X, const Mtx& Y){
    Mtx O = feed_forward(X);
    return cost.accuracy(O, Y);
  }
public:
  FFN1(int iDim, int oDim, size_t iterations, double learning_rate, double regularization_rate, bool display_loss):
    W(Mtx::random(iDim, oDim)), updateW(iDim, oDim), learning_rate(learning_rate),
    regularization_rate(regularization_rate), iterations(iterations), display_loss(display_loss){}

  double train(const Mtx& X, const Mtx& Y){
    for (size_t k = 0; k < iterations; ++k){
      Mtx dW;

      Mtx O = feed_forward(X);
      compute_loss(O, Y, k);
      back_propagation(X, O, Y, dW);
      param_update(dW);
    }

    return test_accuracy(X, Y);
  }
  double test(const Mtx& X, const Mtx& Y){
    return test_accuracy(X, Y);
  }
  Mtx predict(const Mtx& X){
    return feed_forward(X);
  }
  void save(const char* filename){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    int dl = display_loss;

    out.write((char*)&learning_rate, sizeof(double));
    out.write((char*)&regularization_rate, sizeof(double));
    out.write((char*)&iterations, sizeof(size_t));
    out.write((char*)&dl, sizeof(int));
    W.save(out);
  }
  void load(const char* filename){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    int dl;

    in.read((char*)&learning_rate, sizeof(double));
    in.read((char*)&regularization_rate, sizeof(double));
    in.read((char*)&iterations, sizeof(size_t));
    in.read((char*)&dl, sizeof(int));
    W = Mtx(in);

    display_loss = dl;
  }
};

template <typename Cost, typename Activation, typename Regularization, typename ParamUpdate, bool IsDropout>
class FFN {
  std::vector<Mtx> Ws;
  std::vector<Mtx> Hs; //TODO: this does not need to be here
  std::vector<ParamUpdate> updaters;
  Cost cost;
  Activation activation;
  Regularization regularization;
  double learning_rate;
  double regularization_rate;
  size_t iterations;
  bool display_loss;

  Mtx dropout(const Mtx& X){
    Hs.clear();

    Mtx H1 = X * Ws.front();
    activation.function(H1);
    apply_dropout(H1);
    Hs.push_back(H1);
    for (size_t i = 1; i < Ws.size() - 1; ++i){
      Mtx H = Hs.back() * Ws[i];
      activation.function(H);
      apply_dropout(H);
      Hs.push_back(H);
    }
    Mtx O = Hs.back() * Ws.back();
    cost.classification(O);
    return O;
  }

  Mtx feed_forward(const Mtx& X){
    Hs.clear();

    Mtx H1 = X * Ws.front();
    activation.function(H1);
    Hs.push_back(H1);
    for (size_t i = 1; i < Ws.size() - 1; ++i){
      Mtx H = Hs.back() * Ws[i];
      activation.function(H);
      Hs.push_back(H);
    }
    Mtx O = Hs.back() * Ws.back();
    cost.classification(O);
    return O;
  }

  double compute_loss(const Mtx& O, const Mtx& Y, int icount){
    double data_loss = 0., reg_loss = 0.;

    data_loss = cost.loss(O, Y);
    
    for (size_t i = 0; i < Ws.size(); ++i)
      reg_loss += regularization.loss(Ws[i], regularization_rate);

    if (display_loss && icount % (iterations / 100) == 0)
      std::cout << "data: " << data_loss << " reg: " << reg_loss << std::endl;

    return data_loss + reg_loss;
  }

  void back_propagation(const Mtx& X, const Mtx& O, const Mtx& Y, std::vector<Mtx>& dWs){
    Mtx dY = cost.deriviative(O, Y);
    Mtx dWz = Hs.back().transpose() * dY;
    regularization.regularize(dWz, Ws.back(), regularization_rate);
    dWs.push_back(dWz);
    Mtx dH = dY * Ws.back().transpose();
    activation.deriviative(dH, Hs.back());
    for (int i = Ws.size() - 2; i > 0; --i){
      Mtx dW = Hs[i-1].transpose() * dH;
      regularization.regularize(dW, Ws[i], regularization_rate);
      dWs.push_back(dW);
      dH = dH * Ws[i].transpose();
      activation.deriviative(dH, Hs[i-1]);
    }
    Mtx dW1 = X.transpose() * dH;
    regularization.regularize(dW1, Ws.front(), regularization_rate);
    dWs.push_back(dW1);

    //dWs is constructed in reverse
    reverse(dWs.begin(), dWs.end());
  }

  void param_update(const std::vector<Mtx>& dWs){
    for (size_t i = 0; i < Ws.size(); ++i)
      updaters[i].update(Ws[i], dWs[i], learning_rate);
  }

  double test_accuracy(const Mtx& X, const Mtx& Y){
    Mtx O = feed_forward(X);
    return cost.accuracy(O, Y);
  }

public:
  FFN(const std::vector<int>& dims, size_t iterations, double learning_rate, double regularization_rate, bool display_loss):
    learning_rate(learning_rate), regularization_rate(regularization_rate), iterations(iterations), display_loss(display_loss) {
    for (size_t i = 1; i < dims.size(); ++i){
      Ws.push_back(Mtx::random(dims[i-1], dims[i]));
      updaters.push_back(ParamUpdate(dims[i-1], dims[i]));
    }
  }
  double train(const Mtx& X, const Mtx& Y){
    for (size_t k = 0; k < iterations; ++k){
      std::vector<Mtx> dWs;

      Mtx O = IsDropout ? dropout(X) : feed_forward(X);
      compute_loss(O, Y, k);
      back_propagation(X, O, Y, dWs);
      param_update(dWs);
    }

    return test_accuracy(X, Y);
  }

  double test(const Mtx& X, const Mtx& Y){
    return test_accuracy(X, Y);
  }

  Mtx predict(const Mtx& X){
    return feed_forward(X);
  }

  void save(const char* filename){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    size_t wsize = Ws.size();
    int dl = display_loss;

    out.write((char*)&learning_rate, sizeof(double));
    out.write((char*)&regularization_rate, sizeof(double));
    out.write((char*)&iterations, sizeof(size_t));
    out.write((char*)&dl, sizeof(int));
    out.write((char*)&wsize, sizeof(size_t));
    for (size_t i = 0; i < Ws.size(); ++i){
      Ws[i].save(out);
    }
  }

  void load(const char* filename){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    size_t wsize; int dl;
    Ws.clear();

    in.read((char*)&learning_rate, sizeof(double));
    in.read((char*)&regularization_rate, sizeof(double));
    in.read((char*)&iterations, sizeof(size_t));
    in.read((char*)&dl, sizeof(int));
    in.read((char*)&wsize, sizeof(size_t));
    for (size_t i = 0; i < wsize; ++i){
      Mtx mtx(in);
      Ws.emplace_back(std::move(mtx));
    }

    display_loss = dl;
  }
};

} //ML

#endif //ML_FFN

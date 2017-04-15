#ifndef ML_FFN
#define ML_FFN

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include <ffn_util.h>

using Eigen::MatrixXd;

template <typename Cost, typename Regularization, typename ParamUpdate>
class FFN1 {
  MatrixXd W;
  ParamUpdate updateW;
  Cost cost;
  Regularization regularization;
  double learning_rate;
  double regularization_rate;
  size_t iterations;
  bool display_loss;

  MatrixXd feed_forward(const MatrixXd& X){
    MatrixXd O = X * W;
    cost.classification(O);
    return O;
  }
  double compute_loss(const MatrixXd& O, const MatrixXd& Y, size_t icount){
    double data_loss, reg_loss;

    data_loss = cost.loss(O, Y);
    reg_loss = regularization.loss(W, regularization_rate);

    if (display_loss && icount % (iterations / 100) == 0)
      cout << "data: " << data_loss << " reg: " << reg_loss << endl;

    return data_loss + reg_loss;
  }

  void back_propagation(const MatrixXd& X, const MatrixXd& O, const MatrixXd& Y, MatrixXd& dW){
    MatrixXd dY = cost.deriviative(O, Y);
    dW = X.transpose() * dY;
    regularization.regularize(dW, W, regularization_rate);
  }

  void param_update(const MatrixXd& dW){
    updateW.update(W, dW, learning_rate);
  }
  double test_accuracy(const MatrixXd& X, const MatrixXd& Y){
    MatrixXd O = feed_forward(X);
    return cost.accuracy(O, Y);
  }
public:
  FFN1(int iDim, int oDim, size_t iterations, double learning_rate, double regularization_rate, bool display_loss):
    W(random_matrix(iDim, oDim)), updateW(iDim, oDim), learning_rate(learning_rate),
    regularization_rate(regularization_rate), iterations(iterations), display_loss(display_loss){}

  double train(const MatrixXd& X, const MatrixXd& Y){
    for (size_t k = 0; k < iterations; ++k){
      MatrixXd dW;

      MatrixXd O = feed_forward(X);
      compute_loss(O, Y, k);
      back_propagation(X, O, Y, dW);
      param_update(dW);
    }

    return test_accuracy(X, Y);
  }
  double test(const MatrixXd& X, const MatrixXd& Y){
    return test_accuracy(X, Y);
  }
  MatrixXd predict(const MatrixXd& X){
    return feed_forward(X);
  }
  void save(const char* filename){
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    int dl = display_loss;

    out.write((char*)&learning_rate, sizeof(double));
    out.write((char*)&regularization_rate, sizeof(double));
    out.write((char*)&iterations, sizeof(size_t));
    out.write((char*)&dl, sizeof(int));
    out << W;
  }
  void load(const char* filename){
    ifstream in(filename, ios::in | ios::binary);
    int dl;

    in.read((char*)&learning_rate, sizeof(double));
    in.read((char*)&regularization_rate, sizeof(double));
    in.read((char*)&iterations, sizeof(size_t));
    in.read((char*)&dl, sizeof(int));
    in >> W;

    display_loss = dl;
  }
};

template <typename Cost, typename Activation, typename Regularization, typename ParamUpdate, bool IsDropout>
class FFN {
  vector<MatrixXd> Ws;
  vector<MatrixXd> Hs; //TODO: this does not need to be here
  vector<ParamUpdate> updaters;
  Cost cost;
  Activation activation;
  Regularization regularization;
  double learning_rate;
  double regularization_rate;
  size_t iterations;
  bool display_loss;

  MatrixXd dropout(const MatrixXd& X){
    Hs.clear();

    MatrixXd H1 = X * Ws.front();
    activation.function(H1);
    apply_dropout(H1);
    Hs.push_back(H1);
    for (size_t i = 1; i < Ws.size() - 1; ++i){
      MatrixXd H = Hs.back() * Ws[i];
      activation.function(H);
      apply_dropout(H);
      Hs.push_back(H);
    }
    MatrixXd O = Hs.back() * Ws.back();
    cost.classification(O);
    return O;
  }

  MatrixXd feed_forward(const MatrixXd& X){
    Hs.clear();

    MatrixXd H1 = X * Ws.front();
    activation.function(H1);
    Hs.push_back(H1);
    for (size_t i = 1; i < Ws.size() - 1; ++i){
      MatrixXd H = Hs.back() * Ws[i];
      activation.function(H);
      Hs.push_back(H);
    }
    MatrixXd O = Hs.back() * Ws.back();
    cost.classification(O);
    return O;
  }

  double compute_loss(const MatrixXd& O, const MatrixXd& Y, int icount){
    double data_loss = 0., reg_loss = 0.;

    data_loss = cost.loss(O, Y);
    
    for (size_t i = 0; i < Ws.size(); ++i)
      reg_loss += regularization.loss(Ws[i], regularization_rate);

    if (display_loss && icount % (iterations / 100) == 0)
      cout << "data: " << data_loss << " reg: " << reg_loss << endl;

    return data_loss + reg_loss;
  }

  void back_propagation(const MatrixXd& X, const MatrixXd& O, const MatrixXd& Y, vector<MatrixXd>& dWs){
    MatrixXd dY = cost.deriviative(O, Y);
    MatrixXd dWz = Hs.back().transpose() * dY;
    regularization.regularize(dWz, Ws.back(), regularization_rate);
    dWs.push_back(dWz);
    MatrixXd dH = dY * Ws.back().transpose();
    activation.deriviative(dH, Hs.back());
    for (int i = Ws.size() - 2; i > 0; --i){
      MatrixXd dW = Hs[i-1].transpose() * dH;
      regularization.regularize(dW, Ws[i], regularization_rate);
      dWs.push_back(dW);
      dH = dH * Ws[i].transpose();
      activation.deriviative(dH, Hs[i-1]);
    }
    MatrixXd dW1 = X.transpose() * dH;
    regularization.regularize(dW1, Ws.front(), regularization_rate);
    dWs.push_back(dW1);

    //dWs is constructed in reverse
    reverse(dWs.begin(), dWs.end());
  }

  void param_update(const vector<MatrixXd>& dWs){
    for (size_t i = 0; i < Ws.size(); ++i)
      updaters[i].update(Ws[i], dWs[i], learning_rate);
  }

  double test_accuracy(const MatrixXd& X, const MatrixXd& Y){
    MatrixXd O = feed_forward(X);
    return cost.accuracy(O, Y);
  }

public:
  FFN(const vector<int>& dims, size_t iterations, double learning_rate, double regularization_rate, bool display_loss):
    learning_rate(learning_rate), regularization_rate(regularization_rate), iterations(iterations), display_loss(display_loss) {
    for (size_t i = 1; i < dims.size(); ++i){
      Ws.push_back(random_matrix(dims[i-1], dims[i]));
      updaters.push_back(ParamUpdate(dims[i-1], dims[i]));
    }
  }
  double train(const MatrixXd& X, const MatrixXd& Y){
    for (size_t k = 0; k < iterations; ++k){
      vector<MatrixXd> dWs;

      MatrixXd O = IsDropout ? dropout(X) : feed_forward(X);
      compute_loss(O, Y, k);
      back_propagation(X, O, Y, dWs);
      param_update(dWs);
    }

    return test_accuracy(X, Y);
  }

  double test(const MatrixXd& X, const MatrixXd& Y){
    return test_accuracy(X, Y);
  }

  MatrixXd predict(const MatrixXd& X){
    return feed_forward(X);
  }

  void save(const char* filename){
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    size_t wsize = Ws.size();
    int dl = display_loss;

    out.write((char*)&learning_rate, sizeof(double));
    out.write((char*)&regularization_rate, sizeof(double));
    out.write((char*)&iterations, sizeof(size_t));
    out.write((char*)&dl, sizeof(int));
    out.write((char*)&wsize, sizeof(size_t));
    for (size_t i = 0; i < Ws.size(); ++i){
      out << Ws[i];
    }
  }

  void load(const char* filename){
    ifstream in(filename, ios::in | ios::binary);
    size_t wsize; int dl;
    Ws.clear();

    in.read((char*)&learning_rate, sizeof(double));
    in.read((char*)&regularization_rate, sizeof(double));
    in.read((char*)&iterations, sizeof(size_t));
    in.read((char*)&dl, sizeof(int));
    in.read((char*)&wsize, sizeof(size_t));
    for (size_t i = 0; i < wsize; ++i){
      MatrixXd mtx; in >> mtx;
      Ws.emplace_back(std::move(mtx));
    }

    display_loss = dl;
  }
};

#endif

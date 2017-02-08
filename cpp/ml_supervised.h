#ifndef ML_SUPERVISED
#define ML_SUPERVISED

#include <matrix.h>
#include <ml_util.h>

template <typename ACT>
struct Backpropagation: SNN<ACT> {
  Backpropagation(size_t din, size_t dout, size_t dh): SNN<ACT>(din, dout, dh) {}

private:
  void back_propagation(Mtx<double>& X, const Mtx<double>& O, const Mtx<double>& Y, Mtx<double>& dWxh, Mtx<double>& dWho, double reg){
    ACT activation;
    Mtx<double>& Wxh = (*this).Wxh;
    Mtx<double>& Who = (*this).Who;
    Mtx<double>& H = (*this).H;

    Mtx<double> dY = O - Y;
    size_t dYrows = dY.rows();
    dY.foreach([dYrows](double& d){
        d /= (double)dYrows;
    });

    dWho = H.t() * dY;
    H.t(); //revert internal transposition

    Mtx<double> dH = dY * Who.t();
    activation.deriviative(dH, H);
    Who.t(); //revert internal transposition

    dWxh = X.t() * dH;
    X.t(); //revert internal transposition

    //regularizations
    dWho.foreach([reg](double& d, const double& w){
        d += reg * w;
    }, Who);
    dWxh.foreach([reg](double& d, const double& w){
        d += reg * w;
    }, Wxh);
  }

  void compute_loss(const Mtx<double>& O, const Mtx<double>& Y, size_t iteration, double reg, size_t loss_report){
    if (iteration % loss_report != 0) return;

    Mtx<double>& Wxh = static_cast<SNN<ACT>&>(*this).Wxh;
    Mtx<double>& Who = static_cast<SNN<ACT>&>(*this).Who;

    // data loss
    double data_loss = 0.;
    vector<size_t> idxes = Y.maxi(MRow);
    for (size_t i = 0; i < O.rows(); ++i){
      RowRef<double> o = O.row(i);
      data_loss += -1. * log(o[idxes[i]]);
    }
    data_loss /= Y.rows();

    // regularization loss
    double reg_loss = 0.;
    Wxh.foreach([&reg_loss](double& d){
        reg_loss += d * d;
    });
    Who.foreach([&reg_loss](double& d){
        reg_loss += d * d;
    });
    reg_loss *= 0.5 * reg;

    double loss = data_loss + reg_loss;

    cout << "loss: " << loss << " data: " << data_loss << " reg: " << reg_loss << endl;
  }

  void param_update(const Mtx<double>& dWxh, const Mtx<double>& dWho, double step){
    Mtx<double>& Wxh = (*this).Wxh;
    Mtx<double>& Who = (*this).Who;

    //simple update
    Wxh.foreach([step](double& w, const double& d){
        w += -1. * step * d;
    }, dWxh);
    Who.foreach([step](double& w, const double& d){
        w += -1. * step * d;
    }, dWho);
  }

  double compute_accuracy(const Mtx<double>& O, const Mtx<double>& Y){
    vector<size_t> maxo = O.maxi(MRow);
    vector<size_t> maxy = Y.maxi(MRow);
    size_t corrects = 0;
    for (size_t i = 0; i < maxo.size(); ++i)
      if (maxo[i] == maxy[i])
        corrects += 1;

    return (double)corrects / (double)Y.rows();
  }

public:
  double train(Mtx<double>& X, const Mtx<double>& Y, size_t iterations, double step, double reg, size_t loss_report){
    for (size_t k = 0; k < iterations; ++k){
      Mtx<double> dWxh, dWho;

      Mtx<double> O = (*this).feed_forward(X);    //feed forward
      compute_loss(O, Y, k, reg, loss_report);    //compute loss
      back_propagation(X, O, Y, dWxh, dWho, reg); //back propagation
      param_update(dWxh, dWho, step);             //update parameter
    }

    Mtx<double> O = (*this).feed_forward(X);
    return compute_accuracy(O, Y);
  }

  double test(const Mtx<double>& X, const Mtx<double>& Y){
    Mtx<double> O = (*this).feed_forward(X);
    return compute_accuracy(O, Y);
  }

  Mtx<double> predict(const Mtx<double>& x){
    return (*this).feed_forward(x);
  }
};

#endif

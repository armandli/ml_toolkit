#ifndef ML_FUNCTION
#define ML_FUNCTION

#include <ml_matrix.h>
#include <ml_exprtree.h>

namespace ML {

/* Cost Functions */
struct MSE {
  auto loss(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::sum((std::move(O) - std::move(Y)) * (std::move(O) - std::move(Y))) / O.mtx().rows()) {
    return   ML::sum((std::move(O) - std::move(Y)) * (std::move(O) - std::move(Y))) / O.mtx().rows();
  }
  auto accuracy(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::sqrt(ML::sum((std::move(O) - std::move(Y)) * (std::move(O) - std::move(Y)))) * 0.5 / O.mtx().rows()) {
    return   ML::sqrt(ML::sum((std::move(O) - std::move(Y)) * (std::move(O) - std::move(Y)))) * 0.5 / O.mtx().rows();
  }
  auto deriviative(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::isnan0((std::move(O) - std::move(Y)) / Y.mtx().rows())) {
    return   ML::isnan0((std::move(O) - std::move(Y)) / Y.mtx().rows());
  }
  template <typename A>
  A&& classification(MtxBase<A>&& O){
    return static_cast<A&&>(O);
  }
};

struct CrossEntropy {
  auto loss(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::ce_loss(std::move(O), std::move(Y))) {
    return   ML::ce_loss(std::move(O), std::move(Y));
  }
  auto accuracy(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::ce_accuracy(std::move(O), std::move(Y))) {
    return   ML::ce_accuracy(std::move(O), std::move(Y));
  }
  auto deriviative(MtxRef&& O, MtxRef&& Y) ->
    decltype(ML::isnan0((std::move(O) - std::move(Y)) / Y.mtx().rows())) {
    return   ML::isnan0((std::move(O) - std::move(Y)) / Y.mtx().rows());
  }
  template <typename A>
  auto classification(MtxBase<A>&& O) ->
    decltype(ML::softmax(static_cast<A&&>(O))) {
    return   ML::softmax(static_cast<A&&>(O));
  }
};

/* Activation Functions */
struct SigmoidFun {
  template <typename A>
  auto function(MtxBase<A>&& m) ->
    decltype(1. / (1. + ML::exp(static_cast<A&&>(m) - 1.))) {
    return   1. / (1. + ML::exp(static_cast<A&&>(m) - 1.));
  }
  template <typename A, typename B>
  auto deriviative(MtxBase<A>&& dm, MtxBase<B>&& m) ->
    decltype(static_cast<A&&>(dm) * static_cast<B&&>(m) * (1. - static_cast<B&&>(m))) {
    return   static_cast<A&&>(dm) * static_cast<B&&>(m) * (1. - static_cast<B&&>(m));
  }
};

/* Regularization Functions */
struct L2Reg {
  template <typename A>
  auto loss(MtxBase<A>&& W, double reg) ->
    decltype(ML::sum(static_cast<A&&>(W) * static_cast<A&&>(W)) * 0.5 * reg) {
    return   ML::sum(static_cast<A&&>(W) * static_cast<A&&>(W)) * 0.5 * reg;
  }
  template <typename A>
  auto regularize(MtxBase<A>&& dW, MtxRef&& W, Scl&& reg) ->
    decltype(static_cast<A&&>(dW) + std::move(reg) * std::move(W)) {
    return   static_cast<A&&>(dW) + std::move(reg) * std::move(W);
  }
};

/* Parameter Update Functions */
class MomentumUpdate {
  Mtx V;
  double mu;
public:
  MomentumUpdate(size_t rsize, size_t csize, double mu = 0.9): V(rsize, csize), mu(mu) {}
  auto update(MtxRef&& W, MtxRef&& dW, double lrate) ->
    decltype(std::move(W) + MtxRef(V)) {
    V = V * mu - std::move(dW) * lrate;
    return std::move(W) + MtxRef(V);
  }
};

} //ML

#endif//ML_FUNCTION

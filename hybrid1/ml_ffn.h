#ifndef ML_FFN
#define ML_FFN

#include <vector>

#include <ml_matrix.h>
#include <ml_exprtree.h>
#include <ml_arena.h>

namespace ML {

template <typename Cost, typename Regularization, typename ParamUpdate>
class FFN1 {
  Mtx W;
  ParamUpdate updateW;
  Cost cost;
  Regularization regularization;
  MemArena arena;
  //TODO: CUDA Arena
  double learning_rate;
  double regularization_rate;
  size_t iterations;
  bool display_loss;

  void feed_forward(Mtx& O, const Mtx& X){
    O = cost.classification(X ^ W);
  }

  void back_propagation(Mtx& dW, const Mtx& X, const Mtx& O, const Mtx& Y){
    auto dY = cost.deriviative(MtxRef(O), MtxRef(Y));
    auto dW_pre = (~X) ^ std::move(dY);
    dW = regularization.regularize(std::move(dW_pre), MtxRef(W), Scl(regularization_rate));
  }

  void param_update(const Mtx& dW){
    W = updateW.update(MtxRef(W), MtxRef(dW), learning_rate);
  }

  void compute_loss(ReductionResult& data_loss, ReductionResult& reg_loss, const Mtx& O, const Mtx& Y){
    data_loss = cost.loss(O, Y);
    reg_loss = regularization.loss(MtxRef(W), regularization_rate);
  }

  double test_accuracy(const Mtx& X, const Mtx& Y){
    Mtx O;
    feed_forward(O, X);
    ReductionResult accuracy = cost.accuracy(O, Y);
    accuracy.evaluate(arena);
    return accuracy;
  }

public:
  FFN1(size_t iDim, size_t oDim, size_t iterations, double learning_rate, double regularization_rate, bool display_loss):
    W(iDim, oDim, RandomizationType::Gaussian, 0., 0.5), updateW(iDim, oDim), learning_rate(learning_rate), regularization_rate(regularization_rate), 
    iterations(iterations), display_loss(display_loss) {}

  double train(const Mtx& X, const Mtx& Y){
    Mtx O, dW;
    ReductionResult data_loss, reg_loss;
    for (size_t k = 0; k < iterations; ++k){
      feed_forward(O, X);
      if (display_loss && k % (iterations / 100) == 0)
        compute_loss(data_loss, reg_loss, O, Y);
      back_propagation(dW, X, O, Y);
      param_update(dW);
      W.evaluate(arena);

      if (display_loss && k % (iterations / 100) == 0){
        data_loss.evaluate(arena);
        reg_loss.evaluate(arena);
        std::cout << "data: " << data_loss << " " << "reg: " << reg_loss << std::endl;
      }
    }

    return test_accuracy(X, Y);
  }

  double test(const Mtx& X, const Mtx& Y){
    return test_accuracy(X, Y);
  }

  void predict(Mtx& O, const Mtx& X){
    feed_forward(O, X);
    O.evaluate(arena);
  }
};

} // ML

#endif//ML_FFN

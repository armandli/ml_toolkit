#ifndef ML_RL
#define ML_RL

#include <iostream>
#include <ctime>
#include <random>

#include <ml_common.h>
#include <ml_ffn.h>

namespace ML {

struct GameModelIF {
  virtual void init() = 0;
  virtual int get_input_dimension() = 0;
  virtual int get_output_dimension() = 0;
  virtual Mtx to_input() = 0;
  virtual void make_move(int) = 0;
  virtual double get_reward() = 0;
  virtual bool is_terminal_state() = 0;

  virtual std::ostream& report_stat(std::ostream& out){
    return out;
  }
};

template <typename NNModel>
class RFL {
  GameModelIF& game;
  NNModel& model;
  double gamma;

  struct Experience {
    Mtx Xbatch;
    Mtx Ybatch;
    int iDim;
    int oDim;
    int eidx;
    bool experience_filled;

    Experience(int iDim, int oDim, int batchsize):
      Xbatch(Mtx::zeros(batchsize, iDim)), Ybatch(Mtx::zeros(batchsize, oDim)),
      iDim(iDim), oDim(oDim), eidx(0), experience_filled(false)
    {}

    void update(Mtx& X, Mtx& Y){
      Xbatch.block(eidx, 0, 1, iDim) = X;
      Ybatch.block(eidx, 0, 1, oDim) = Y;

      if (++eidx >= Xbatch.rows()){
        eidx = 0;
        experience_filled = true;
      }
    }

    bool run_calibration(){
      return experience_filled;
    }
  };

  bool make_random_move(double epsilon){
    std::uniform_real_distribution<double> dist(0., 1.);
    std::default_random_engine& eng = get_default_random_engine();

    if (dist(eng) < epsilon) return true;
    else                     return false;
  }
  int random_action(int size){
    std::uniform_int_distribution<int> dist(0, size - 1);
    std::default_random_engine& eng = get_default_random_engine();

    return dist(eng);
  }
public:
  RFL(GameModelIF& game, NNModel& model, double g = 0.975):
    game(game), model(model), gamma(g)
  {}

  void train(size_t epoch, size_t batchsize = 300, size_t report_frequency = 10){
    int iDim = game.get_input_dimension();
    int oDim = game.get_output_dimension();
    double epsilon = 1.;
    Experience exp(iDim, oDim, batchsize);

    for (size_t k = 0; k < epoch; ++k){
      game.init();
      while (not game.is_terminal_state()){
        Mtx X = game.to_input();
        Mtx O = model.predict(X);

        int action;
        if (make_random_move(epsilon))
          action = random_action(oDim);
        else {
          std::vector<DimV> mcoeffs = O.max_coeff(MRow);
          action = mcoeffs[0].idx;
        }
        game.make_move(action);
        double reward = game.get_reward();
        Mtx Xn = game.to_input();
        Mtx On = model.predict(Xn);
        std::vector<DimV> rcoeffs = On.max_coeff(MRow);
        double maxq = rcoeffs[0].val;
        double update;
        if (game.is_terminal_state()) update = reward;
        else                          update = reward + gamma * maxq;
        O(0, action) = update;

        exp.update(X, O);
        if (exp.run_calibration())
          model.train(exp.Xbatch, exp.Ybatch);
      }
      if (epsilon > .1) epsilon -= 1 / epoch;

      if (k % report_frequency == 0){
        std::cout << "epoch: " << k << ": ";
        game.report_stat(std::cout);
      }
    }
  }
};

} //ML

#endif //ML_RL

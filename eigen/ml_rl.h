#ifndef ML_RL
#define ML_RL

#include <iostream>
#include <ctime>
#include <random>

#include <ml_util.h>

#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

struct GameModelIF {
  virtual void init() = 0;
  virtual int get_input_dimension() = 0;
  virtual int get_output_dimension() = 0;
  virtual MatrixXd to_input() = 0;
  virtual void make_move(int) = 0;
  virtual double get_reward() = 0;
  virtual bool is_terminal_state() = 0;

  virtual ostream& report_stat(ostream& out){
    return out;
  }
};

template <typename NNModel>
class RFL {
  GameModelIF& game;
  NNModel& model;
  double gamma;

  struct Experience {
    MatrixXd Xbatch;
    MatrixXd Ybatch;
    int iDim;
    int oDim;
    int eidx;
    bool experience_filled;

    Experience(int iDim, int oDim, int batchsize):
      Xbatch(MatrixXd::Zero(batchsize, iDim)), Ybatch(MatrixXd::Zero(batchsize, oDim)),
      iDim(iDim), oDim(oDim), eidx(0), experience_filled(false)
    {}

    void update(MatrixXd& X, MatrixXd& Y){
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
    uniform_real_distribution<double> dist(0., 1.);
    default_random_engine& eng = get_default_random_engine();

    if (dist(eng) < epsilon) return true;
    else                     return false;
  }
  int random_action(int size){
    uniform_int_distribution<int> dist(0, size - 1);
    default_random_engine& eng = get_default_random_engine();

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
        MatrixXd X = game.to_input();
        MatrixXd O = model.predict(X);

        int action;
        if (make_random_move(epsilon))
          action = random_action(oDim);
        else {
          MatrixXd::Index idx;
          O.row(0).maxCoeff(&idx);
          action = idx;
        }
        game.make_move(action);
        double reward = game.get_reward();
        MatrixXd Xn = game.to_input();
        MatrixXd On = model.predict(Xn);
        double maxq = On.maxCoeff();
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
        cout << "epoch: " << k << ": ";
        game.report_stat(cout);
      }
    }
  }
};

#endif //ML_RL

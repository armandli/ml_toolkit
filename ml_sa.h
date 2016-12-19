#ifndef ML_SA
#define ML_SA

#include <matrix.h>
#include <ml_util.h>

template <typename ACT>
struct SimulatedAnealing: SNN<ACT> {
  SimulatedAnealing(size_t din, size_t dout, size_t dh): SNN<ACT>(din, dout, dh) {}

private:

public:
  double train(const Mtx<double>& x, const Mtx<double>& y){ //TODO
    //TODO
  }

  double test(const Mtx<double>& x, const Mtx<double>& y){
    //TODO
  }

  Mtx<double> prdict(const Mtx<double>& x){
    //TODO
  }
};

#endif

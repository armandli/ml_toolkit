#ifndef ML_UTIL
#define ML_UTIL

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

#endif

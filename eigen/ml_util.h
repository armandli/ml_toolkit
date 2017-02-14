#ifndef ML_UTIL
#define ML_UTIL

#include <ctime>
#include <random>
#include <set>

void validation_split(const MatrixXd& X, const MatrixXd& Y, double testPercentage, MatrixXd& trainX, MatrixXd& trainY, MatrixXd& testX, MatrixXd& testY){
  uniform_int_distribution<size_t> dist(0, X.rows() - 1);
  default_random_engine eng(time(0));
  size_t testn = X.rows() * testPercentage;
  set<size_t> test_rows;

  while (test_rows.size() < testn)
    test_rows.insert(dist(eng));

  trainX = MatrixXd(X.rows() - testn, X.cols());
  trainY = MatrixXd(Y.rows() - testn, Y.cols());
  testX  = MatrixXd(testn, X.cols());
  testY  = MatrixXd(testn, Y.cols());

  for (size_t i = 0, traini = 0, testi = 0; i < X.rows(); ++i){
    if (test_rows.find(i) == test_rows.end()){
      trainX.block(traini, 0, 1, X.cols()) = X.block(i, 0, 1, X.cols());
      trainY.block(traini, 0, 1, Y.cols()) = Y.block(i, 0, 1, Y.cols());
      traini++;
    } else {
      testX.block(testi, 0, 1, X.cols()) = X.block(i, 0, 1, X.cols());
      testY.block(testi, 0, 1, Y.cols()) = Y.block(i, 0, 1, Y.cols());
      testi++;
    }
  }
}

MatrixXd add_bias(const MatrixXd& X){
  MatrixXd ret(X.rows(), X.cols() + 1);
  MatrixXd ones = MatrixXd::Ones(X.rows(), 1);
  ret.block(0, 0, X.rows(), X.cols()) = X;
  ret.block(0, X.cols(), X.rows(), 1) = ones;
  return ret;
}

#endif

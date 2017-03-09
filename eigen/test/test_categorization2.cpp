#include <cstdlib>
#include <set>
#include <vector>
#include <string>
#include <ctime>
#include <random>
#include <iostream>
#include <fstream>

#include <ml_ffn.h>

//test using MSVM cost function

using namespace std;
using Eigen::MatrixXd;

void create_input(const vector<double> v, MatrixXd& trainX, MatrixXd& trainY, MatrixXd& testX, MatrixXd& testY){
  int total_rows = v.size() / 3;

  uniform_int_distribution<int> dist(0, total_rows - 1);
  default_random_engine eng(time(0));
  set<int> test_rows;

  while (test_rows.size() < total_rows / 10)
    test_rows.insert(dist(eng));

  trainX = MatrixXd(total_rows - test_rows.size(), 3);
  trainY = MatrixXd(total_rows - test_rows.size(), 3);
  testX = MatrixXd(test_rows.size(), 3);
  testY = MatrixXd(test_rows.size(), 3);

  for (int i = 0, trainR = 0, testR = 0; i < total_rows; ++i)
    if (test_rows.find(i) == test_rows.end()){
      trainX(trainR, 0) = v[i * 3];
      trainX(trainR, 1) = v[i * 3 + 1];
      trainX(trainR, 2) = 1.;
      trainY(trainR, 0) = trainY(trainR, 1) = trainY(trainR, 2) = 0.;
      trainY(trainR, (int)v[i * 3 + 2]) = 1.;
      trainR++;
    } else {
      testX(testR, 0) = v[i * 3];
      testX(testR, 1) = v[i * 3 + 1];
      testX(testR, 2) = 1.;
      testY(testR, 0) = testY(testR, 1) = testY(testR, 2) = 0.;
      testY(testR, (int)v[i * 3 + 2]) = 1.;
      testR++;
    }
}

int main(){
  ifstream file("../../data/spiral_dataset.txt");
  vector<double> v; double tmp;
  while (file >> tmp){
    v.push_back(tmp);
  }
  file.close();

  MatrixXd trainX, trainY, testX, testY;
  create_input(v, trainX, trainY, testX, testY);

  vector<int> dims;
  dims.push_back(trainX.cols());
  dims.push_back(128);
  dims.push_back(trainY.cols());
  FFN<MSVM, ReluFun, L2Reg, AdamUpdate, false> net(dims, 4000, 0.08, 0.001, true);
  double train_accuracy = net.train(trainX, trainY);
  double test_accuracy = net.test(testX, testY);

  cout << "training accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;
}

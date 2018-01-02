#include <cstdlib>
#include <set>
#include <vector>
#include <string>
#include <ctime>
#include <random>
#include <iostream>
#include <fstream>

#include <ml_function.h>
#include <ml_ffn.h>
#include <ml_ssa.h>
#include <ml_mem_codegen.h>

using ML::Mtx;
using namespace std;

void create_input(const vector<double> v, Mtx& trainX, Mtx& trainY, Mtx& testX, Mtx& testY){
  size_t total_rows = v.size() / 3;

  uniform_int_distribution<int> dist(0, total_rows - 1);
  set<int> test_rows;

  while (test_rows.size() < total_rows / 10)
    test_rows.insert(dist(ML::get_default_random_engine()));

  trainX.init(total_rows - test_rows.size(), 3);
  trainY.init(total_rows - test_rows.size(), 3);
  testX.init(test_rows.size(), 3);
  testY.init(test_rows.size(), 3);

  for (size_t i = 0, trainR = 0, testR = 0; i < total_rows; ++i)
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

using Model = ML::FFN2<ML::CrossEntropy, ML::ReluFun, ML::L2Reg, ML::AdamUpdate>;

int main(){
  ifstream file("../../data/spiral_dataset.txt");
  vector<double> v; double tmp;
  while (file >> tmp){
    v.push_back(tmp);
  }
  file.close();

  Mtx trainX, trainY, testX, testY;
  create_input(v, trainX, trainY, testX, testY);

  vector<size_t> dims;
  dims.push_back(trainX.cols());
  dims.push_back(128);
  dims.push_back(trainY.cols());
  Model net(dims, 2000, 0.08, 0.001, true);
  double train_accuracy = net.train(trainX, trainY);
  double test_accuracy = net.test(testX, testY);

  //TODO: the result here is suspicious, need to investigate
  cout << "training accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;
}

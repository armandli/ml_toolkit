#include <cstdlib>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <ctime>
#include <random>
#include <iostream>

#include <csv_reader.h>
#include <ml_common.h>
#include <ml_ffn.h>

using namespace std;

using ML::Mtx;
using Model = ML::FFN1<ML::MSE, ML::L2Reg, ML::NesterovUpdate>;

struct ReadPolynomial {
  Mtx trainX;
  Mtx trainY;
  Mtx testX;
  Mtx testY;

  bool operator()(const map<string, const char*>& m, int){
    vector<double> v;
    v.push_back(atof(m.at(string("a"))));
    v.push_back(atof(m.at(string("b"))));
    v.push_back(atof(m.at(string("Y"))));
    values.emplace_back(move(v));
    return true;
  }
  void genXY(){
    uniform_int_distribution<int> dist(0, values.size() - 1);
    set<int> test_rows;

    while (test_rows.size() < values.size() / 5)
      test_rows.insert(dist(ML::get_default_random_engine()));

    const int C = 2;
    const int trainR = values.size() - test_rows.size();
    const int testR = test_rows.size();

    trainX = Mtx(trainR, C + 1);
    trainY = Mtx(trainR, 1);
    testX  = Mtx(testR, C + 1);
    testY  = Mtx(testR, 1);
    for (int i = 0, traini = 0, testi = 0; i < values.size(); ++i){
      if (test_rows.find(i) == test_rows.end()){
        for (int j = 0; j < C; ++j)
          trainX(traini, j) = values[i][j];
        trainX(traini, 2)  = 1.; //add bias
        trainY(traini, 0) = values[i][2];
        traini++;
      } else {
        for (int j = 0; j < C; ++j)
          testX(testi, j) = values[i][j];
        testX(testi, 2) = 1.;
        testY(testi, 0) = values[i][2];
        testi++;
      }
    }
  }
private:
  vector<vector<double>> values;
};

Mtx formula_compute(const Mtx& X){
//formula: 0.724295a + 2.8905b + -1.51562
  vector<double> vw = {0.724295, 2.8905, -1.51562};
  Mtx W(3, 1, vw);
  Mtx Y = X * W;
  return Y;
}

int main(){
  csv_reader reader("../../data/linear2d_5000.csv");

  if (not reader.is_open()) return 1;

  ReadPolynomial read_function;
  reader.process(read_function);
  read_function.genXY();

  Model network(3, 1, 10000, 0.0001, 0.000001, true);

//  vector<int> dims;
//  dims.push_back(6);
//  dims.push_back(64);
//  dims.push_back(1);
//  ML::FFN<ML::MSE, ML::TanhFun, ML::L2Reg, ML::AdamUpdate, false> network(dims, 10000, 0.08, 0.001, true);
  
  double train_accuracy = network.train(read_function.trainX, read_function.trainY);
  double test_accuracy = network.test(read_function.testX, read_function.testY);

  cout << "train accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;

  Mtx O = network.predict(read_function.testX);
  Mtx T = formula_compute(read_function.testX);
  Mtx D = O - T;
  D.binary_expr([](double d, double t){
      return abs(d) / abs(t);
  }, T);
//  cout << D << endl;

  vector<double> sums = D.sum(ML::MAll);
  cout << "average: " << sums[0] / D.rows() << endl;
}

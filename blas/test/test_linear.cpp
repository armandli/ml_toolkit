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
using Model = ML::FFN<ML::MSE, ML::ReluFun, ML::L2Reg, ML::AdamUpdate, false>;

struct ReadPolynomial {
  Mtx trainX;
  Mtx trainY;
  Mtx testX;
  Mtx testY;

  bool operator()(const map<string, const char*>& m, int){
    vector<double> v;
    v.push_back(atof(m.at(string("a"))));
    v.push_back(atof(m.at(string("b"))));
    v.push_back(atof(m.at(string("c"))));
    v.push_back(atof(m.at(string("d"))));
    v.push_back(atof(m.at(string("e"))));
    v.push_back(atof(m.at(string("Y"))));
    values.emplace_back(move(v));
    return true;
  }
  void genXY(){
    uniform_int_distribution<int> dist(0, values.size() - 1);
    set<int> test_rows;

    while (test_rows.size() < values.size() / 5)
      test_rows.insert(dist(ML::get_default_random_engine()));

    const int C = 5;
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
        trainX(traini, 5)  = 1.; //add bias
        trainY(traini, 0) = values[i][5];
        traini++;
      } else {
        for (int j = 0; j < C; ++j)
          testX(testi, j) = values[i][j];
        testX(testi, 5) = 1.;
        testY(testi, 0) = values[i][5];
        testi++;
      }
    }
  }
private:
  vector<vector<double>> values;
};

Mtx formula_compute(const Mtx& X){
//formula: -0.57375a + -1.41037b + -1.98335c + -1.90862d + -0.573063e + 0.753815
  vector<double> vw = {-0.57375, -1.41037, -1.98335, -1.90862, -0.573063, 0.753815};
  Mtx W(6, 1, vw);
  Mtx Y = X * W;
  return Y;
}

int main(){
  csv_reader reader("../../data/linear_5000.csv");

  if (not reader.is_open()) return 1;

  ReadPolynomial read_function;
  reader.process(read_function);
  read_function.genXY();

  vector<int> dims;
  dims.push_back(6);
  dims.push_back(64);
  dims.push_back(1);
  Model network(dims, 10000, 0.08, 0.001, true);
  
  double train_accuracy = network.train(read_function.trainX, read_function.trainY);
  double test_accuracy = network.test(read_function.testX, read_function.testY);

  cout << "train accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;

  Mtx O = network.predict(read_function.testX);
  Mtx T = formula_compute(read_function.testX);
  Mtx D = O - T;
  D.binary_expr([](double d, double t){
      return abs(d) / abs(t);
  }, T);
  cout << D << endl;

  double sum = D.sum();
  cout << "average: " << sum / D.rows() << endl;
}

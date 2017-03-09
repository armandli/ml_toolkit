#include <cstdlib>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <ctime>
#include <random>
#include <iostream>

#include <csv_reader.h>
#include <ml_ffn.h>

#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

struct ReadPolynomial {
  MatrixXd trainX;
  MatrixXd trainY;
  MatrixXd testX;
  MatrixXd testY;

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
    default_random_engine eng(time(0));
    set<int> test_rows;

    while (test_rows.size() < values.size() / 5)
      test_rows.insert(dist(eng));

    const int C = 5;
    const int trainR = values.size() - test_rows.size();
    const int testR = test_rows.size();

    trainX = MatrixXd(trainR, C + 1);
    trainY = MatrixXd(trainR, 1);
    testX  = MatrixXd(testR, C + 1);
    testY  = MatrixXd(testR, 1);
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

int main(){
  set<csv_reader::Column> columns;
  columns.insert((csv_reader::Column){"a", true});
  columns.insert((csv_reader::Column){"b", true});
  columns.insert((csv_reader::Column){"c", true});
  columns.insert((csv_reader::Column){"d", true});
  columns.insert((csv_reader::Column){"e", true});
  columns.insert((csv_reader::Column){"Y", true});
  csv_reader reader("../../data/polynomial_5000.csv", columns);

  if (not reader.is_open()) return 1;

  ReadPolynomial read_function;
  reader.process(read_function);
  read_function.genXY();

  vector<int> layers;
  layers.push_back(6);
  layers.push_back(64);
  layers.push_back(1);
  FFN<MSE, ReluFun, L2Reg, AdamUpdate, false> network(layers, 200000, 0.08, 0.001, true);

  double train_accuracy = network.train(read_function.trainX, read_function.trainY);
  double test_accuracy = network.test(read_function.testX, read_function.testY);

  cout << "train accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;
}

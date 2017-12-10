#include <iostream>

#include <csv_reader.h>
#include <ml_function.h>
#include <ml_ffn.h>
#include <ml_ssa.h>
#include <ml_mem_codegen.h>

using ML::Mtx;
using namespace std;

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

    trainX.init(trainR, C + 1);
    trainY.init(trainR, 1);
    testX.init(testR, C + 1);
    testY.init(testR, 1);
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

void formula_compute(Mtx& T, Mtx& W, const Mtx& X){
//formula: 0.724295a + 2.8905b + -1.51562
  vector<double> vw = {0.724295, 2.8905, -1.51562};
  W.init(3, 1);
  W(0, 0) = vw[0];
  W(1, 0) = vw[1];
  W(2, 0) = vw[2];
  T = X ^ W;
}


using Model = ML::FFN1<ML::MSE, ML::L2Reg, ML::MomentumUpdate>;

int main(){
  csv_reader reader("../../data/linear2d_5000.csv");

  if (not reader.is_open()) return 1;

  ReadPolynomial read_function;
  reader.process(read_function);
  read_function.genXY();

  Model model(3, 1, 10000, 0.0001, 0.000001, true);

  double train_accuracy = model.train(read_function.trainX, read_function.trainY);
  double test_accuracy = model.test(read_function.testX, read_function.testY);

  cout << "train accuracy: " << train_accuracy << " test accuracy: " << test_accuracy << endl;

  ML::MemArena arena;
  Mtx O, T, W;
  model.predict(O, read_function.testX);
  formula_compute(T, W, read_function.testX);
  Mtx D = ML::abs(O) / ML::abs(T);
  ML::ReductionResult r = ML::sum(D) / D.rows();
  r.evaluate(arena);
  std::cout << "average:"<< r << std::endl;
}

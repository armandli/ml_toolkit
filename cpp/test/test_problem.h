#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <ctime>
#include <matrix.h>
using namespace std;

void read_input(const string& filename, Mtx<double>& TrainX, Mtx<double>& TrainY, Mtx<double>& TestX, Mtx<double>& TestY){
  srand((int)time(0));

  ifstream in(filename.c_str());
  vector<double> v; double tmp;
  while (in >> tmp)
    v.push_back(tmp);

  size_t total = v.size() / 3;
  set<size_t> test_indexes;
  while (test_indexes.size() < total / 5){
    size_t idx = rand() % total;
    test_indexes.insert(idx);
  }

  vector<double> trainx, trainy, testx, testy;
  for (size_t i = 0; i < total; ++i){
    vector<double>* tx = nullptr, *ty = nullptr;
    if (test_indexes.find(i) == test_indexes.end()){
      tx = &trainx; ty = &trainy;
    } else {
      tx = &testx; ty = &testy;
    }
    tx->push_back(v[i * 3]);
    tx->push_back(v[i * 3 + 1]);
    tx->push_back(1.);
    if (v[i * 3 + 2] == 0.){
      ty->push_back(1.);
      ty->push_back(0.);
      ty->push_back(0.);
    } else if (v[i * 3 + 2] == 1.){
      ty->push_back(0.);
      ty->push_back(1.);
      ty->push_back(0.);
    } else {
      ty->push_back(0.);
      ty->push_back(0.);
      ty->push_back(1.);
    }
  }

  TrainX = Mtx<double>(3, trainx.size() / 3, trainx); TrainX.t(); //transpose
  TrainY = Mtx<double>(3, trainy.size() / 3, trainy); TrainY.t();
  TestX = Mtx<double>(3, testx.size() / 3, testx); TestX.t();
  TestY = Mtx<double>(3, testy.size() / 3, testy); TestY.t();
}

#include <vector>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>
#include <random>
using namespace std;

// generate N rows of data based on a polynomial function of K input parameter and 1 output Y
// adding a gaussian noise of standard deviation L

const int DIM = 3;

vector<double> gen_params(int K){
  uniform_real_distribution<double> dist(-3., 3.);
  default_random_engine eng(time(0));
  vector<double> ret;

  for (int i = 0; i < K; ++i)
    for (int j = 0; j < DIM; ++j)
      ret.push_back(dist(eng));
  ret.push_back(dist(eng)); //offset
  return ret;
}

vector<vector<double>> gen_inputs(int N, int K){
  uniform_real_distribution<double> dist(-100., 100);
  default_random_engine eng(time(0));
  vector<vector<double>> ret;

  for (int i = 0; i < N; ++i){
    vector<double> r;
    for (int j = 0; j < K; ++j)
      r.push_back(dist(eng));
    ret.emplace_back(move(r));
  }
  return ret;
}

void print_formula(int K, const vector<double>& params){
  char input_char = 'a';
  bool is_first = true;

  for (int i = 0; i < K; ++i, ++input_char)
    for (int j = 0; j < DIM; ++j){
      if (is_first) is_first = false;
      else          cout << "+ ";
      cout << params[j + i * DIM] << input_char;
      switch (j){
        case 0: cout << "^3 "; break;
        case 1: cout << "^2 "; break;
        case 2: cout << " ";   break;
      }
    }
  cout << "+ " << params[params.size() - 1] << endl;
}

double run_formula(const vector<double>& params, const vector<double>& X, int K, double L){
  normal_distribution<double> dist(0.0, L);
  default_random_engine eng(time(0));
  double Y = 0.;

  for (int i = 0; i < K; ++i)
    for (int j = 0; j < DIM; ++j)
      switch (j){
        case 0: Y += params[j + i * DIM] * pow(X[i], 3.); break;
        case 1: Y += params[j + i * DIM] * pow(X[i], 2.); break;
        case 2: Y += params[j + i * DIM] * X[i];          break;
      }
  Y += params[params.size() - 1]; //add offset
  Y += dist(eng) * Y; //add noise

  return Y;
}

void gen_data(int N, int K, double L){
  normal_distribution<double> dist(0.0, L);
  default_random_engine eng(time(0));

  vector<double> params = gen_params(K);
  vector<vector<double>> inputs = gen_inputs(N, K);

  cout << "formula: ";
  print_formula(K, params);

  bool is_first = true;
  char input_char = 'a';
  for (int i = 0; i < K; ++i, ++input_char){
    if (is_first) is_first = false;
    else          cout << ",";
    cout << input_char;
  }
  cout << ",Y" << endl;

  for (int i = 0; i < N; ++i){
    is_first = true;
    for (int j = 0; j < K; ++j){
      if (is_first) is_first = false;
      else          cout << ",";
      cout << inputs[i][j];
    }
    cout << "," << run_formula(params, inputs[i], K, L) << endl;
  }
}

int main(int argc, char* argv[]){
  if (argc != 4){
    cout << "Usage: " << argv[0] << " [N data point] [K input dim] [L noise sdv]" << endl;
    exit(1);
  }

  int n = atoi(argv[1]);
  int k = atoi(argv[2]);
  double l = atof(argv[3]);

  gen_data(n, k, l);
}

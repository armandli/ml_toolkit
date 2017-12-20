#include <Eigen/Dense>

#include <ctime>
#include <iostream>

using namespace std;

using Eigen::MatrixXd;

#define SZ 2048

int main(){
  clock_t sum = 0;
  MatrixXd a = MatrixXd::Random(SZ, SZ);
  MatrixXd b = MatrixXd::Random(SZ, SZ);
  for (size_t i = 0; i < 100; ++i){
    clock_t start = clock();
    MatrixXd c = a * b;
    sum += clock() - start;
    cout << c(0, 0) << endl;
  }
  cout << "Time: " << (sum / (double)(CLOCKS_PER_SEC / 1000)) << " ms" << endl;
}

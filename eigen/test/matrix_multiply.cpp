#include <Eigen/Dense>

#include <ctime>
#include <iostream>

using namespace std;

using Eigen::MatrixXd;

#define SZ 2048

int main(){
  MatrixXd a = MatrixXd::Random(SZ, SZ);
  MatrixXd b = MatrixXd::Random(SZ, SZ);

  clock_t start = clock();
  MatrixXd c = a * b;
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cout << c(0, 0) << endl;
}

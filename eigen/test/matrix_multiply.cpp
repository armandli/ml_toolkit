#include <Eigen/Dense>

#include <ctime>
#include <iostream>

using namespace std;

using Eigen::MatrixXd;

#define SZ 32

int main(){
  MatrixXd a = MatrixXd::Random(SZ, SZ);
  MatrixXd b = MatrixXd::Random(SZ, SZ);

  clock_t start = clock();
  MatrixXd c = a * b;
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool iszero = false;
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (c(i, j) == 0.){
        iszero = true;
      }

  if (iszero) cout << "zero found" << endl;
  else        cout << "zero not found" << endl;
}

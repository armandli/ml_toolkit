#include <Eigen/Dense>

#include <ctime>
#include <iostream>

using namespace std;

using Eigen::MatrixXd;

#define SZ 2048

int main(){
  MatrixXd a = MatrixXd(SZ, SZ);
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      a(i, j) = (double)i * (double)j;

  clock_t start = clock();
  MatrixXd b = a.transpose();
  cout << "Time: " << (clock() - start) << " us" << endl;

  bool is_correct = true;
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (a(i, j) != b(j, i)){
        is_correct = false;
        break;
      }   

  if (is_correct) cout << "correct!" << endl;
  else            cout << "not correct!" << endl;
}

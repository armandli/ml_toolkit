#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 2048

Mtx transpose(const Mtx& m){
  return m.transpose();
}

int main(){
  Mtx a(SZ, SZ);
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      a(i, j) = (double)i * (double)j;

  clock_t start = clock();
  Mtx b = transpose(a);
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

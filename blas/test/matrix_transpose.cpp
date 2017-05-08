#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 4096

Mtx transpose(const Mtx& m){
  return m.transpose();
}

int main(){
  Mtx a = Mtx::random(SZ, SZ);

  clock_t start = clock();
  Mtx b = transpose(a);
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool iszero = false;
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (b(i, j) == 0.){
        iszero = true;
      }   

  if (iszero) cout << "zero found" << endl;
  else        cout << "zero not found" << endl;
}

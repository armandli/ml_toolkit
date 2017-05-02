#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 32

int main(){
  Mtx a = Mtx::random(SZ, SZ);
  Mtx b = Mtx::random(SZ, SZ);

  clock_t start = clock();
  Mtx c = a * b;
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

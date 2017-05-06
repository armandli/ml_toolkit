#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 4096

int main(){
  Mtx a = Mtx::random(SZ, SZ);

  clock_t start = clock();
  a.transpose();
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool iszero = false;
  for (size_t i = 0; i < SZ; ++i)
    for (size_t j = 0; j < SZ; ++j)
      if (a(i, j) == 0.){
        iszero = true;
      }   

  if (iszero) cout << "zero found" << endl;
  else        cout << "zero not found" << endl;
}

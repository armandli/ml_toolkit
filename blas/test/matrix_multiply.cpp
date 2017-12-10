#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 2048

int main(){
  Mtx a = Mtx::random(SZ, SZ);
  Mtx b = Mtx::random(SZ, SZ);

  clock_t start = clock();
  Mtx c = a * b;
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cout << c(0, 0) << endl;
}

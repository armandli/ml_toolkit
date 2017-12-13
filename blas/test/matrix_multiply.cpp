#include <ctime>
#include <iostream>

#include <ml_matrix.h>

using namespace std;
using namespace ML;

#define SZ 2048

int main(){
  Mtx a = Mtx::random(SZ, SZ);
  Mtx b = Mtx::random(SZ, SZ);

  clock_t sum = 0;
  for (size_t i = 0; i < 100; ++i){
    clock_t start = clock();
    Mtx c = a * b;
    sum += clock() - start;
    cout << c(0, 0) << endl;
  }
  cout << "Time: " << (sum / (double)(CLOCKS_PER_SEC / 1000)) << " ms" << endl;
}

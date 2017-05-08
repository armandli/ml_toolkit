#include <matrix.h>

#include <ctime>
#include <iostream>

using namespace std;

#define SZ 64
using tt = double;

int main(){
  Mtx<tt> a = Mtx<double>::random(SZ, SZ);

  clock_t start = clock();
  a.t();
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

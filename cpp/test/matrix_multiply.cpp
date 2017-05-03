#include <matrix.h>

#include <ctime>
#include <iostream>

using namespace std;

#define SZ 2048
using tt = double;

int main(){
  Mtx<tt> a = Mtx<double>::random(SZ, SZ);
  Mtx<tt> b = Mtx<double>::random(SZ, SZ);

  clock_t start = clock();
  Mtx<tt> c = a.flip() * b;
//  Mtx<tt> c = a * b;
  cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

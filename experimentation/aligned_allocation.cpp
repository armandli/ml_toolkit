#include <iostream>
#include <stdlib.h>

using namespace std;

void use_posix_memalign(double** d1, double** d2){
  posix_memalign((void**)d1, 16, sizeof(double) * 16);
  posix_memalign((void**)d2, 32, sizeof(double) * 32);

  cout << (size_t)*d1 << endl;
  cout << (size_t)*d2 << endl;
}

void use_aligned_alloc(double** d1, double** d2){
  *d1 = (double*)aligned_alloc(16, sizeof(double) * 16);
  *d2 = (double*)aligned_alloc(32, sizeof(double) * 32);

  cout << (size_t)*d1 << endl;
  cout << (size_t)*d2 << endl;
}

int main(){
  typedef double* doublep;
  doublep d1, d2, d3, d4;

  use_posix_memalign(&d1, &d2);
  use_aligned_alloc(&d3, &d4);

  free(d1);
  free(d2);
  free(d3);
  free(d4);
}

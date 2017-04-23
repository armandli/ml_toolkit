#include <cblas.h>
#include <cstdio>

//dgemm routine calculates:
//C = alpha * A * B + beta * C
//dgemm routine parameter explanation:
//  cblas_dgemm(
//    CblasRowMajor, - indicate matrices are stored in row major order
//    CblasNoTrans,  - indicate A should not be transposed before multiplication
//    CblasNoTrans,  - indicate B should not be transposed before multiplication
//    m, n, k,       - indicate size of matrices, A = m rows by k columns, B = k rows by n columns, C = m rows by n columns
//    alpha,         - multiplier on A
//    A,             - array used to store A
//    k,             - leading dimension of array A, for row major this is same as number of columns
//    B,             - array used to store B
//    n,             - leading dimension of array B, or number of elements between successive rows (row major)
//    beta,          - beta multiplier on C
//    C,             - array used to store C
//    n,             - leading dimension of array C
//  )

int main(){
  double A[6] = {1.0,2.0,1.0,
                -3.0,4.0,-1.0};         
  double B[6] = {1.0,2.0,
                 1.0,-3.0,
                 4.0,-1.0};  
  double C[9] = {.5,.5,.5,
                 .5,.5,.5,
                 .5,.5,.5}; 
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);

  for(int i=0; i < 3; i++){
    for (int j = 0; j < 3; ++j)
      printf("%lf ", C[i * 3 + j]);
    printf("\n");
  }
}

#include <cblas.h>
#include <iostream>

using namespace std;

int main()
{
   const enum CBLAS_ORDER Order= CblasRowMajor; //行主序	
   const enum CBLAS_TRANSPOSE TransA = CblasNoTrans; 
   const enum CBLAS_TRANSPOSE TransB = CblasNoTrans; 
   const int M = 4; //A 行数
   const int N = 2; //B 列数　
   const int K = 3; //A 列数，B行数
   const float alpha = 1; 
   const float beta = 0; 
   const int lda = K; //A的列
   const int ldb = N; //B的列
   const int ldc = N; //C的列
   const float A[M*K] = {1,2,3,4,5,6,7,8,9,10,11,12}; //4行３列
   const float B[K*N] = {5,4,3,2,1,0};//3,2
   float C[M*N]; 
   cblas_sgemm(
   CblasRowMajor, 
   TransA, 
   TransB, 
   M, N, K, 
   alpha, 
   A, lda, 
   B, ldb, 
   beta, C, 
   ldc);
   for(int i=0; i<M; i++)
   {
     for(int j=0; j<N; j++)
     {
       cout<<C[i*N+j]<<" ";
     }
     cout<<endl;
   }
   return 0;
}


// Microbenchmark: OpenBLAS dgesv vs dposv on an SPD n x n system, nrhs RHS.
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int c, char**v){int n=atoi(v[1]),r=atoi(v[2]);
 double*S=malloc(8.0*n*n),*A=malloc(8.0*n*n),*B0=malloc(8.0*n*r),*B=malloc(8.0*n*r);int*p=malloc(4*n);
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)S[i*n+j]=(i==j)?n:1.0/(1+i+j);
 for(int i=0;i<n*r;i++)B0[i]=i%7;
 for(int k=0;k<2;k++){double best[2]={1e9,1e9};for(int it=0;it<21;it++){
  memcpy(A,S,8.0*n*n);memcpy(B,B0,8.0*n*r);double t=now();
  if(k==0)scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR,n,r,A,n,p,B,r);else scipy_LAPACKE_dposv(LAPACK_ROW_MAJOR,'L',n,r,A,n,B,r);
  t=now()-t;if(t<best[0])best[0]=t;}
  printf("%s n=%d nrhs=%d min_s=%.3e\n",k?"posv":"gesv",n,r,best[0]);}}

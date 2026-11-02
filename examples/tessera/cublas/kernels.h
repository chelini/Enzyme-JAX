#ifndef KERNELS_H
#define KERNELS_H
#include <cublas_v2.h>
void gram(cublasHandle_t h, int n, int m, const double *X, double *G);
void opt(cublasHandle_t h, int n, int m, const double *A, const double *W, double *AW, double *M);
void tri(cublasHandle_t h, int n, int m, const double *L, const double *U, const double *B, double *Y, double *Z);
void diag(cublasHandle_t h, int n, int m, const double *D, const double *E, const double *B, double *Y, double *Z);
#endif

// The four kernels, straightforward cuBLAS C. Only this file goes through the
// Tessera pipeline: the harness (bench.c) allocates device memory, which the
// pipeline's CPU backend would turn into host malloc.
#include "tessera_cublas.h"

#ifdef NO_FACTS
#define FACT(s)
#else
#define FACT(s) __attribute__((tessera_guarantees(s)));
#endif

#define N_ CUBLAS_OP_N
#define T_ CUBLAS_OP_T

// ---------------------------------------------------------------- kernels
__attribute__((noinline)) void gram(cublasHandle_t h, int n, int m,
                                    const double *X, double *G) {
  tl_dgemm(h, T_, N_, m, m, n, 1.0, X, n, X, n, 0.0, G, m);
}

// W is n x n diagonal. AW is an m x n workspace.
__attribute__((noinline)) void opt(cublasHandle_t h, int n, int m,
                                   const double *A, const double *W,
                                   double *AW, double *M) {
  FACT("diagonal(W)")
  tl_dgemm(h, N_, N_, m, n, n, 1.0, A, m, W, n, 0.0, AW, m);
  tl_dgemm(h, N_, T_, m, m, n, 1.0, AW, m, A, m, 0.0, M, m);
}

__attribute__((noinline)) void tri(cublasHandle_t h, int n, int m,
                                   const double *L, const double *U,
                                   const double *B, double *Y, double *Z) {
  FACT("triangular_lower(L)")
  FACT("triangular_upper(U)")
  tl_dgemm(h, N_, N_, n, m, n, 1.0, U, n, B, n, 0.0, Y, n);
  tl_dgemm(h, N_, N_, n, m, n, 1.0, L, n, B, n, 0.0, Z, n);
}

__attribute__((noinline)) void diag(cublasHandle_t h, int n, int m,
                                    const double *D, const double *E,
                                    const double *B, double *Y, double *Z) {
  FACT("diagonal(D)")
  FACT("diagonal(E)")
  tl_dgemm(h, N_, N_, n, m, n, 1.0, D, n, B, n, 0.0, Y, n);
  tl_dgemm(h, N_, N_, n, m, m, 1.0, B, n, E, m, 0.0, Z, n);
}


// Linnea example problems written as straightforward C over OpenBLAS.
// The kernels call only general routines (GEMM, GESV); the facts a Linnea
// user would declare (SPD, diagonal, triangular) are stated with Tessera.
//
//   ols   Linnea Example01  b = (X^T X)^-1 X^T y,            X full rank
//   gls   Linnea Example02  z = (X^T S^-1 X)^-1 X^T S^-1 y,  S SPD
//   opt   Linnea Example03  x = W (A^T (A W A^T)^-1 b - c),  W diagonal SPD
//   tri   triangular solve and product: x = L^-1 B, Y = U B
//   diag  diagonal system: x = D^-1 B
//
// Usage: bench <kernel> <n> <m> <reps>   (prints median seconds and error)
#include "tessera_openblas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// A fact a Linnea user would declare; -DNO_FACTS drops them all.
#ifdef NO_FACTS
#define FACT(s)
#else
#define FACT(s) __attribute__((tessera_guarantees(s)));
#endif

#define RM CblasRowMajor
#define NT CblasNoTrans
#define TR CblasTrans

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static unsigned long long rng = 88172645463325252ull;
static double urand(void) {
  rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
  return (double)(rng >> 11) / 9007199254740992.0 - 0.5;
}
static double *alloc(size_t n) { return (double *)malloc(n * sizeof(double)); }
static double *rand_mat(int r, int c) {
  double *a = alloc((size_t)r * c);
  for (size_t i = 0; i < (size_t)r * c; ++i)
    a[i] = urand();
  return a;
}

// ---------------------------------------------------------------- kernels
__attribute__((noinline)) void ols(int n, int m, const double *X,
                                   const double *y, double *b, double *G,
                                   int *ipiv) {
  scipy_cblas_dgemm(RM, TR, NT, m, m, n, 1.0, X, m, X, m, 0.0, G, m);
  FACT("SPD(G)")
  scipy_cblas_dgemv(RM, TR, n, m, 1.0, X, m, y, 1, 0.0, b, 1);
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, G, m, ipiv, b, 1);
}

// R is n x (m+1) holding [X y]; it becomes S^-1 [X y]. M (m x m) and z (m)
// receive the normal equations X^T S^-1 X z = X^T S^-1 y, and z the solution.
__attribute__((noinline)) void gls(int n, int m, const double *X,
                                   const double *S, const double *y,
                                   double *Sc, double *R, double *M, double *z,
                                   int *ipiv) {
  int m1 = m + 1;
  memcpy(Sc, S, (size_t)n * n * sizeof(double));
  FACT("SPD(Sc)")
  for (int i = 0; i < n; ++i) {
    memcpy(R + (size_t)i * m1, X + (size_t)i * m, m * sizeof(double));
    R[(size_t)i * m1 + m] = y[i];
  }
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, m1, Sc, n, ipiv, R, m1);
  scipy_cblas_dgemm(RM, TR, NT, m, m, n, 1.0, X, m, R, m1, 0.0, M, m);
  FACT("SPD(M)")
  scipy_cblas_dgemv(RM, TR, n, m, 1.0, X, m, R + m, m1, 0.0, z, 1);
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, M, m, ipiv, z, 1);
}

// A is m x n, W n x n diagonal. AW and Mm are workspaces.
__attribute__((noinline)) void opt(int n, int m, const double *A,
                                   const double *W, const double *bv,
                                   const double *c, double *x, double *AW,
                                   double *Mm, double *t, double *u,
                                   int *ipiv) {
  FACT("diagonal(W)")
  scipy_cblas_dgemm(RM, NT, NT, m, n, n, 1.0, A, n, W, n, 0.0, AW, n);
  scipy_cblas_dgemm(RM, NT, TR, m, m, n, 1.0, AW, n, A, n, 0.0, Mm, m);
  FACT("SPD(Mm)")
  memcpy(t, bv, m * sizeof(double));
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, Mm, m, ipiv, t, 1);
  memcpy(u, c, n * sizeof(double));
  scipy_cblas_dgemv(RM, TR, m, n, 1.0, A, n, t, 1, -1.0, u, 1);
  for (int i = 0; i < n; ++i)
    x[i] = W[(size_t)i * n + i] * u[i];
}

// L lower and U upper triangular (n x n), B n x m. X = L^-1 B, Y = U B.
__attribute__((noinline)) void tri(int n, int m, double *L, const double *U,
                                   const double *B, double *X, double *Y,
                                   int *ipiv) {
  FACT("triangular_lower(L)")
  FACT("triangular_upper(U)")
  memcpy(X, B, (size_t)n * m * sizeof(double));
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, m, L, n, ipiv, X, m);
  scipy_cblas_dgemm(RM, NT, NT, n, m, n, 1.0, U, n, B, m, 0.0, Y, m);
}

__attribute__((noinline)) void diag(int n, int m, double *D, const double *B,
                                    double *X, int *ipiv) {
  FACT("diagonal(D)")
  memcpy(X, B, (size_t)n * m * sizeof(double));
  scipy_LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, m, D, n, ipiv, X, m);
}

// ------------------------------------------------------- reference checks
// Naive loops, independent of BLAS, for the relative error of each result.
static double rel(const double *got, const double *want, size_t len) {
  double num = 0, den = 0;
  for (size_t i = 0; i < len; ++i) {
    num += (got[i] - want[i]) * (got[i] - want[i]);
    den += want[i] * want[i];
  }
  return sqrt(num / den);
}
// Residual ||A x - b|| / (||A|| ||x||) for row-major A (n x n) and x (n x m).
static double residual(int n, int m, const double *A, const double *x,
                       const double *b) {
  double num = 0, na = 0, nx = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      double s = -b[(size_t)i * m + j];
      for (int k = 0; k < n; ++k)
        s += A[(size_t)i * n + k] * x[(size_t)k * m + j];
      num += s * s;
    }
  for (size_t i = 0; i < (size_t)n * n; ++i)
    na += A[i] * A[i];
  for (size_t i = 0; i < (size_t)n * m; ++i)
    nx += x[i] * x[i];
  return sqrt(num) / (sqrt(na) * sqrt(nx));
}
static double *gram(int n, int m, const double *X) { // X^T X, m x m
  double *G = calloc((size_t)m * m, sizeof(double));
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        G[(size_t)i * m + j] += X[(size_t)k * m + i] * X[(size_t)k * m + j];
  return G;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s ols|gls|opt|tri|diag n m reps\n", argv[0]);
    return 2;
  }
  const char *k = argv[1];
  int n = atoi(argv[2]), m = atoi(argv[3]), reps = atoi(argv[4]);
  double times[64];
  double err = 0;
  int *ipiv = malloc(sizeof(int) * (n > m ? n : m) + 8);
#define TIME(call)                                                             \
  for (int r = 0; r < reps; ++r) {                                             \
    double t0 = now();                                                         \
    call;                                                                      \
    times[r] = now() - t0;                                                     \
  }

  if (!strcmp(k, "ols")) { // n rows, m columns
    double *X = rand_mat(n, m), *y = rand_mat(n, 1);
    double *b = alloc(m), *G = alloc((size_t)m * m);
    TIME(ols(n, m, X, y, b, G, ipiv));
    double *G0 = gram(n, m, X), *rhs = calloc(m, sizeof(double));
    for (int r = 0; r < n; ++r)
      for (int j = 0; j < m; ++j)
        rhs[j] += X[(size_t)r * m + j] * y[r];
    err = residual(m, 1, G0, b, rhs);
  } else if (!strcmp(k, "gls")) {
    double *X = rand_mat(n, m), *y = rand_mat(n, 1), *Z = rand_mat(n, n);
    double *S = alloc((size_t)n * n);
    for (int i = 0; i < n; ++i) // S = Z Z^T / n + I: SPD
      for (int j = 0; j < n; ++j) {
        double s = (i == j);
        for (int l = 0; l < n; ++l)
          s += Z[(size_t)i * n + l] * Z[(size_t)j * n + l] / n;
        S[(size_t)i * n + j] = s;
      }
    double *Sc = alloc((size_t)n * n), *R = alloc((size_t)n * (m + 1));
    double *M = alloc((size_t)m * m), *z = alloc(m);
    TIME(gls(n, m, X, S, y, Sc, R, M, z, ipiv));
    // Check S R = [X y] for the first solve and the normal equations for z.
    double *XY = alloc((size_t)n * (m + 1));
    for (int i = 0; i < n; ++i) {
      memcpy(XY + (size_t)i * (m + 1), X + (size_t)i * m, m * sizeof(double));
      XY[(size_t)i * (m + 1) + m] = y[i];
    }
    double e1 = residual(n, m + 1, S, R, XY);
    double *Ms = alloc((size_t)m * m), *rhs = alloc(m);
    for (int i = 0; i < m; ++i) { // recompute X^T R independently
      for (int j = 0; j <= m; ++j) {
        double s = 0;
        for (int l = 0; l < n; ++l)
          s += X[(size_t)l * m + i] * R[(size_t)l * (m + 1) + j];
        if (j < m)
          Ms[(size_t)i * m + j] = s;
        else
          rhs[i] = s;
      }
    }
    double e2 = residual(m, 1, Ms, z, rhs);
    err = e1 > e2 ? e1 : e2;
  } else if (!strcmp(k, "opt")) { // A is m x n with m < n
    double *A = rand_mat(m, n), *bv = rand_mat(m, 1), *c = rand_mat(n, 1);
    double *W = calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; ++i)
      W[(size_t)i * n + i] = 1.0 + (urand() + 0.5);
    double *x = alloc(n), *AW = alloc((size_t)m * n), *Mm = alloc((size_t)m * m);
    double *t = alloc(m), *u = alloc(n);
    TIME(opt(n, m, A, W, bv, c, x, AW, Mm, t, u, ipiv));
    // Reference: M = A W A^T, check M t = b, then x = W (A^T t - c).
    double *M0 = calloc((size_t)m * m, sizeof(double)), *x0 = alloc(n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        for (int l = 0; l < n; ++l)
          M0[(size_t)i * m + j] += A[(size_t)i * n + l] *
                                   W[(size_t)l * n + l] * A[(size_t)j * n + l];
    double e1 = residual(m, 1, M0, t, bv);
    for (int l = 0; l < n; ++l) {
      double s = -c[l];
      for (int i = 0; i < m; ++i)
        s += A[(size_t)i * n + l] * t[i];
      x0[l] = W[(size_t)l * n + l] * s;
    }
    double e2 = rel(x, x0, n);
    err = e1 > e2 ? e1 : e2;
  } else if (!strcmp(k, "tri")) {
    double *L = rand_mat(n, n), *U = rand_mat(n, n), *B = rand_mat(n, m);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        if (j > i)
          L[(size_t)i * n + j] = 0;
        if (j < i)
          U[(size_t)i * n + j] = 0;
        if (i == j)
          L[(size_t)i * n + j] = U[(size_t)i * n + j] = 2.0 + urand();
        else
          L[(size_t)i * n + j] /= n, U[(size_t)i * n + j] /= n;
      }
    double *L0 = alloc((size_t)n * n);
    memcpy(L0, L, (size_t)n * n * sizeof(double));
    double *Lc = alloc((size_t)n * n), *X = alloc((size_t)n * m),
           *Y = alloc((size_t)n * m);
    double ts = 0;
    for (int r = 0; r < reps; ++r) {
      memcpy(Lc, L0, (size_t)n * n * sizeof(double)); // gesv may factor L
      double t0 = now();
      tri(n, m, Lc, U, B, X, Y, ipiv);
      times[r] = now() - t0;
      (void)ts;
    }
    double e1 = residual(n, m, L0, X, B);
    double *Y0 = calloc((size_t)n * m, sizeof(double));
    for (int i = 0; i < n; ++i)
      for (int l = i; l < n; ++l)
        for (int j = 0; j < m; ++j)
          Y0[(size_t)i * m + j] += U[(size_t)i * n + l] * B[(size_t)l * m + j];
    double e2 = rel(Y, Y0, (size_t)n * m);
    err = e1 > e2 ? e1 : e2;
  } else if (!strcmp(k, "diag")) {
    double *D = calloc((size_t)n * n, sizeof(double)), *B = rand_mat(n, m);
    for (int i = 0; i < n; ++i)
      D[(size_t)i * n + i] = 1.0 + (urand() + 0.5);
    double *D0 = alloc((size_t)n * n), *X = alloc((size_t)n * m);
    memcpy(D0, D, (size_t)n * n * sizeof(double));
    for (int r = 0; r < reps; ++r) {
      memcpy(D, D0, (size_t)n * n * sizeof(double));
      double t0 = now();
      diag(n, m, D, B, X, ipiv);
      times[r] = now() - t0;
    }
    err = residual(n, m, D0, X, B);
  } else {
    fprintf(stderr, "unknown kernel %s\n", k);
    return 2;
  }
  qsort(times, reps, sizeof(double), cmp);
  printf("%s n=%d m=%d median_s=%.6e err=%.3e\n", k, n, m, times[reps / 2],
         err);
  return 0;
}

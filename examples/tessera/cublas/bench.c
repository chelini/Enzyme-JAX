// Harness for the cuBLAS kernels in kernels.c (Linnea-style problems written
// as straightforward C). The kernels
// call only GEMM; the facts a Linnea user would declare (triangular, diagonal)
// are stated with Tessera. All matrices are column-major device buffers.
//
//   gram  G = X^T X                    X is n x m          (SYRK)
//   opt   M = (A W) A^T                A is m x n, W n x n diagonal  (DGMM)
//   tri   Y = U B and Z = L B          L, U triangular n x n, B n x m (TRMM)
//   diag  Y = D B and Z = B E          D n x n, E m x m diagonal      (DGMM)
//
// Usage: bench <kernel> <n> <m> <reps>   (prints median seconds and error)
#include "kernels.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_ CUBLAS_OP_N
#define T_ CUBLAS_OP_T

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
#define CK(x)                                                                  \
  do {                                                                         \
    int s_ = (int)(x);                                                         \
    if (s_) {                                                                  \
      fprintf(stderr, "%s:%d: error %d\n", __FILE__, __LINE__, s_);            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
static double *host(size_t n) { return (double *)malloc(n * sizeof(double)); }
static double *dev(size_t n) {
  double *p;
  CK(cudaMalloc((void **)&p, n * sizeof(double)));
  return p;
}
static double *up(const double *h, size_t n) { // host -> new device buffer
  double *d = dev(n);
  CK(cudaMemcpy(d, h, n * sizeof(double), cudaMemcpyHostToDevice));
  return d;
}
static double *down(const double *d, size_t n) {
  double *h = host(n);
  CK(cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyDeviceToHost));
  return h;
}
static double *rand_mat(int r, int c) {
  double *a = host((size_t)r * c);
  for (size_t i = 0; i < (size_t)r * c; ++i)
    a[i] = urand();
  return a;
}
#define AT(a, ld, i, j) ((a)[(i) + (size_t)(j) * (ld)])

// ------------------------------------------------------- reference checks
// Sampled entries of the product, by independent host loops: the relative
// error is the worst |got - want| / (|row| |col|) over `samples` entries.
#define SAMPLES 256
static double sample_err(int m, int n, int k, const double *A, int lda,
                         int ta, const double *B, int ldb, int tb,
                         const double *C, int ldc) {
  double worst = 0;
  for (int s = 0; s < SAMPLES; ++s) {
    int i = (int)(urand() * m + m / 2.0) % m, j = (int)(urand() * n + n / 2.0) % n;
    if (s < 8) // include corners and the diagonal region
      i = s % 2 ? m - 1 - s / 2 : s / 2, j = s % 4 < 2 ? i % n : n - 1 - s / 2;
    double sum = 0, na = 0, nb = 0;
    for (int l = 0; l < k; ++l) {
      double a = ta ? AT(A, lda, l, i) : AT(A, lda, i, l);
      double b = tb ? AT(B, ldb, j, l) : AT(B, ldb, l, j);
      sum += a * b, na += a * a, nb += b * b;
    }
    double e = fabs(AT(C, ldc, i, j) - sum) / (sqrt(na) * sqrt(nb) + 1e-300);
    if (e > worst)
      worst = e;
  }
  return worst;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double max2(double a, double b) { return a > b ? a : b; }

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s gram|opt|tri|diag n m reps\n", argv[0]);
    return 2;
  }
  const char *k = argv[1];
  int n = atoi(argv[2]), m = atoi(argv[3]), reps = atoi(argv[4]);
  double times[64], err = 0;
  cublasHandle_t h;
  CK(cublasCreate(&h));
#define TIME(call)                                                             \
  call; /* warm-up: library and kernel load */                                \
  CK(cudaDeviceSynchronize());                                                 \
  for (int r = 0; r < reps; ++r) {                                             \
    CK(cudaDeviceSynchronize());                                               \
    double t0 = now();                                                         \
    call;                                                                      \
    CK(cudaDeviceSynchronize());                                               \
    times[r] = now() - t0;                                                     \
  }

  if (!strcmp(k, "gram")) { // X is n x m
    double *Xh = rand_mat(n, m), *X = up(Xh, (size_t)n * m), *G = dev((size_t)m * m);
    CK(cudaMemset(G, 0, (size_t)m * m * sizeof(double)));
    TIME(gram(h, n, m, X, G));
    double *Gh = down(G, (size_t)m * m);
    err = sample_err(m, m, n, Xh, n, 1, Xh, n, 0, Gh, m);
  } else if (!strcmp(k, "opt")) { // A is m x n
    double *Ah = rand_mat(m, n), *Wh = calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; ++i)
      AT(Wh, n, i, i) = 1.0 + (urand() + 0.5);
    double *A = up(Ah, (size_t)m * n), *W = up(Wh, (size_t)n * n);
    double *AW = dev((size_t)m * n), *M = dev((size_t)m * m);
    CK(cudaMemset(M, 0, (size_t)m * m * sizeof(double)));
    TIME(opt(h, n, m, A, W, AW, M));
    double *AWh = down(AW, (size_t)m * n), *Mh = down(M, (size_t)m * m);
    double e1 = sample_err(m, n, n, Ah, m, 0, Wh, n, 0, AWh, m);
    double e2 = sample_err(m, m, n, AWh, m, 0, Ah, m, 1, Mh, m);
    err = max2(e1, e2);
  } else if (!strcmp(k, "tri")) { // B is n x m
    double *Lh = rand_mat(n, n), *Uh = rand_mat(n, n), *Bh = rand_mat(n, m);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        if (i < j)
          AT(Lh, n, i, j) = 0;
        if (i > j)
          AT(Uh, n, i, j) = 0;
      }
    double *L = up(Lh, (size_t)n * n), *U = up(Uh, (size_t)n * n);
    double *B = up(Bh, (size_t)n * m), *Y = dev((size_t)n * m), *Z = dev((size_t)n * m);
    TIME(tri(h, n, m, L, U, B, Y, Z));
    double *Yh = down(Y, (size_t)n * m), *Zh = down(Z, (size_t)n * m);
    err = max2(sample_err(n, m, n, Uh, n, 0, Bh, n, 0, Yh, n),
               sample_err(n, m, n, Lh, n, 0, Bh, n, 0, Zh, n));
  } else if (!strcmp(k, "diag")) { // D is n x n, E is m x m
    double *Dh = calloc((size_t)n * n, sizeof(double));
    double *Eh = calloc((size_t)m * m, sizeof(double)), *Bh = rand_mat(n, m);
    for (int i = 0; i < n; ++i)
      AT(Dh, n, i, i) = 1.0 + (urand() + 0.5);
    for (int i = 0; i < m; ++i)
      AT(Eh, m, i, i) = 1.0 + (urand() + 0.5);
    double *D = up(Dh, (size_t)n * n), *E = up(Eh, (size_t)m * m);
    double *B = up(Bh, (size_t)n * m), *Y = dev((size_t)n * m), *Z = dev((size_t)n * m);
    TIME(diag(h, n, m, D, E, B, Y, Z));
    double *Yh = down(Y, (size_t)n * m), *Zh = down(Z, (size_t)n * m);
    err = max2(sample_err(n, m, n, Dh, n, 0, Bh, n, 0, Yh, n),
               sample_err(n, m, m, Bh, n, 0, Eh, m, 0, Zh, n));
  } else {
    fprintf(stderr, "unknown kernel %s\n", k);
    return 2;
  }
  qsort(times, reps, sizeof(double), cmp);
  printf("%s n=%d m=%d median_s=%.6e err=%.3e\n", k, n, m, times[reps / 2], err);
  return 0;
}

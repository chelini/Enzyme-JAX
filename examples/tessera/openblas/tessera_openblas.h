// Tessera annotations and Linnea-derived rewrite rules for OpenBLAS.
//
// The declarations below re-declare existing CBLAS/LAPACKE entry points of
// OpenBLAS (scipy-openblas32 build, symbols carry a `scipy_` prefix) with
// Tessera annotations. No OpenBLAS source is modified. Applications include
// this header instead of (after) <cblas.h>/<lapacke.h>.
//
// Every pointer argument is a handle: the matrices have run-time sizes, so
// they are not lifted into SSA values. Rules match a single call and are
// conditioned on properties of its handles.
//
// Contract assumed by the gesv rules: the caller uses only the solution in
// `b`; the factorization left in `a` and the pivots in `ipiv` are scratch.
#ifndef TESSERA_OPENBLAS_H
#define TESSERA_OPENBLAS_H

#include <cblas.h>
#include <lapacke.h>

// CBLAS enumerators, as integers usable in rules.
//   101 RowMajor  111 NoTrans  112 Trans  121 Upper  122 Lower
//   131 NonUnit   141 Left     142 Right
// LAPACK characters: 'L' = 76, 'U' = 85, 'N' = 78, 'A' = 65.

// --- LAPACK solvers -------------------------------------------------------
__attribute__((tessera_op(
    "lapack.gesv(layout, n, nrhs, a, lda, ipiv, b, ldb)")))
lapack_int scipy_LAPACKE_dgesv(int layout, lapack_int n, lapack_int nrhs,
                               double *a, lapack_int lda, lapack_int *ipiv,
                               double *b, lapack_int ldb);

__attribute__((tessera_op(
    "lapack.posv(layout, uplo, n, nrhs, a, lda, b, ldb)")))
lapack_int scipy_LAPACKE_dposv(int layout, char uplo, lapack_int n,
                               lapack_int nrhs, double *a, lapack_int lda,
                               double *b, lapack_int ldb);

__attribute__((tessera_op(
    "lapack.trtrs(layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb)")))
lapack_int scipy_LAPACKE_dtrtrs(int layout, char uplo, char trans, char diag,
                                lapack_int n, lapack_int nrhs, const double *a,
                                lapack_int lda, double *b, lapack_int ldb);

__attribute__((tessera_op(
    "lapack.tbtrs(layout, uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb)")))
lapack_int scipy_LAPACKE_dtbtrs(int layout, char uplo, char trans, char diag,
                                lapack_int n, lapack_int kd, lapack_int nrhs,
                                const double *ab, lapack_int ldab, double *b,
                                lapack_int ldb);

__attribute__((tessera_op(
    "lapack.lacpy(layout, uplo, m, n, a, lda, b, ldb)")))
lapack_int scipy_LAPACKE_dlacpy(int layout, char uplo, lapack_int m,
                                lapack_int n, const double *a, lapack_int lda,
                                double *b, lapack_int ldb);

// --- BLAS-3 products ------------------------------------------------------
__attribute__((tessera_op("blas.gemm(layout, ta, tb, m, n, k, alpha, a, lda, "
                          "b, ldb, beta, c, ldc)"),
               tessera_readonly("A", "B")))
void scipy_cblas_dgemm(const enum CBLAS_ORDER Order,
                       const enum CBLAS_TRANSPOSE TransA,
                       const enum CBLAS_TRANSPOSE TransB, const blasint M,
                       const blasint N, const blasint K, const double alpha,
                       const double *A, const blasint lda, const double *B,
                       const blasint ldb, const double beta, double *C,
                       const blasint ldc);

__attribute__((tessera_op(
    "blas.syrk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)")))
void scipy_cblas_dsyrk(const enum CBLAS_ORDER Order,
                       const enum CBLAS_UPLO Uplo,
                       const enum CBLAS_TRANSPOSE Trans, const blasint N,
                       const blasint K, const double alpha, const double *A,
                       const blasint lda, const double beta, double *C,
                       const blasint ldc);

__attribute__((tessera_op("blas.trmm(layout, side, uplo, ta, diag, m, n, "
                          "alpha, a, lda, b, ldb)")))
void scipy_cblas_dtrmm(const enum CBLAS_ORDER Order,
                       const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                       const enum CBLAS_TRANSPOSE TransA,
                       const enum CBLAS_DIAG Diag, const blasint M,
                       const blasint N, const double alpha, const double *A,
                       const blasint lda, double *B, const blasint ldb);

// --- Two O(n^2) helpers OpenBLAS lacks ------------------------------------
// tl_symmetrize: mirror the lower triangle of a row-major n x n matrix into
// its upper triangle (BLAS syrk writes one triangle only).
// tl_dgmm: C = diag(x) * A (side = 141) or A * diag(x) (side = 142), with the
// diagonal read at stride incx; the interface follows cuBLAS cublasDgmm.
__attribute__((tessera_op("tl.symmetrize(n, c, ldc)"), noinline))
static void tl_symmetrize(int n, double *c, int ldc) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j)
      c[j * ldc + i] = c[i * ldc + j];
}

__attribute__((tessera_op(
                   "tl.dgmm(side, m, n, alpha, a, lda, x, incx, c, ldc)"),
               noinline))
static void tl_dgmm(int side, int m, int n, double alpha, const double *a,
                    int lda, const double *x, int incx, double *c, int ldc) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      c[i * ldc + j] =
          alpha * a[i * lda + j] * x[(side == 141 ? i : j) * incx];
}

// tl_diagsv: B := diag(d)^-1 B for row-major B (n x nrhs), d at stride incd.
// Returns 0 like LAPACKE, or i + 1 if d[i] is zero.
__attribute__((tessera_op("tl.diagsv(n, nrhs, d, incd, b, ldb)"), noinline))
static int tl_diagsv(int n, int nrhs, const double *d, int incd, double *b,
                     int ldb) {
  for (int i = 0; i < n; ++i) {
    double di = d[i * incd];
    if (di == 0.0)
      return i + 1;
    for (int j = 0; j < nrhs; ++j)
      b[i * ldb + j] /= di;
  }
  return 0;
}

#ifndef TESSERA_NO_RULES
// --- Rules ----------------------------------------------------------------
// L1. SPD system: LU (gesv) -> Cholesky (posv).            Linnea: POSV/POTRF
#pragma optimize "if SPD(a), lapack.gesv(layout, n, nrhs, a, lda, ipiv, b, ldb) -> lapack.posv(layout, 76, n, nrhs, a, lda, b, ldb)"

// L2. Triangular system: LU -> triangular solve.            Linnea: TRSM
#pragma optimize "if triangular_lower(a), lapack.gesv(layout, n, nrhs, a, lda, ipiv, b, ldb) -> lapack.trtrs(layout, 76, 78, 78, n, nrhs, a, lda, b, ldb)"
#pragma optimize "if triangular_upper(a), lapack.gesv(layout, n, nrhs, a, lda, ipiv, b, ldb) -> lapack.trtrs(layout, 85, 78, 78, n, nrhs, a, lda, b, ldb)"

// L3. Gram product A^T A: GEMM -> SYRK + mirror.            Linnea: SYRK
#pragma optimize "if layout == 101 && ta == 112 && tb == 111 && lda == ldb && beta == 0.0, blas.gemm(layout, ta, tb, m, n, k, alpha, a, lda, a, ldb, beta, c, ldc) -> blas.syrk(layout, 122, 112, n, k, alpha, a, lda, beta, c, ldc); tl.symmetrize(n, c, ldc)"

// L4. Triangular factor in a product: GEMM -> copy + TRMM.  Linnea: TRMM
#pragma optimize "if triangular_upper(a) && ta == 111 && tb == 111 && m == k && beta == 0.0, blas.gemm(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> lapack.lacpy(layout, 65, m, n, b, ldb, c, ldc); blas.trmm(layout, 141, 121, 111, 131, m, n, alpha, a, lda, c, ldc)"

// L5. Diagonal factor in a product: GEMM -> scaling.        Linnea: DIAGSCAL
#pragma optimize "if diagonal(b) && layout == 101 && ta == 111 && tb == 111 && k == n && beta == 0.0, blas.gemm(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> tl.dgmm(142, m, n, alpha, a, lda, b, ldb + 1, c, ldc)"

// L6. Diagonal system: LU -> diagonal solve.                Linnea: DIAGSV
// (A first version targeted LAPACKE_dtbtrs with kd = 0 and ldab = lda + 1.
// That is wrong for row-major LAPACKE, whose band layout stores the diagonal
// contiguously; the numerical check caught it.)
#pragma optimize "if diagonal(a) && layout == 101, lapack.gesv(layout, n, nrhs, a, lda, ipiv, b, ldb) -> tl.diagsv(n, nrhs, a, lda + 1, b, ldb)"

#endif // TESSERA_NO_RULES

#endif

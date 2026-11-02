// Tessera annotations and Linnea-derived rewrite rules for cuBLAS.
//
// Same idea as tessera_openblas.h: a header re-declares cuBLAS entry points
// with Tessera annotations, and #pragma optimize rules swap a general routine
// for a cheaper specialized one. No cuBLAS source is modified.
//
// Two differences from the CBLAS header:
//  * cuBLAS takes alpha and beta by pointer, so a rule cannot read their
//    values. The tl_* wrappers below take them by value and call cuBLAS with
//    host pointer mode. Applications call the wrappers.
//  * cuBLAS is column-major only, so no rule tests a layout argument.
//
// Every device pointer is a handle. Rules match a single call and are
// conditioned on properties of its handles.
#ifndef TESSERA_CUBLAS_H
#define TESSERA_CUBLAS_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

// cuBLAS enumerators, as integers usable in rules.
//   op:   N = 0, T = 1        fill: Lower = 0, Upper = 1
//   side: Left = 0, Right = 1 diag: NonUnit = 0, Unit = 1

// --- BLAS-3 products (scalars by value) ------------------------------------
__attribute__((tessera_op("cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, b, "
                          "ldb, beta, c, ldc)"),
               tessera_readonly("a", "b"), noinline))
static cublasStatus_t tl_dgemm(cublasHandle_t h, cublasOperation_t ta,
                               cublasOperation_t tb, int m, int n, int k,
                               double alpha, const double *a, int lda,
                               const double *b, int ldb, double beta,
                               double *c, int ldc) {
  return cublasDgemm(h, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c,
                     ldc);
}

__attribute__((tessera_op("cublas.syrk(h, uplo, trans, n, k, alpha, a, lda, "
                          "beta, c, ldc)"),
               noinline))
static cublasStatus_t tl_dsyrk(cublasHandle_t h, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               double alpha, const double *a, int lda,
                               double beta, double *c, int ldc) {
  return cublasDsyrk(h, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
}

// Out-of-place triangular product C = alpha * op(A) * B (cuBLAS 8+).
__attribute__((tessera_op("cublas.trmm(h, side, uplo, ta, diag, m, n, alpha, "
                          "a, lda, b, ldb, c, ldc)"),
               noinline))
static cublasStatus_t tl_dtrmm(cublasHandle_t h, cublasSideMode_t side,
                               cublasFillMode_t uplo, cublasOperation_t ta,
                               cublasDiagType_t diag, int m, int n,
                               double alpha, const double *a, int lda,
                               const double *b, int ldb, double *c, int ldc) {
  return cublasDtrmm(h, side, uplo, ta, diag, m, n, &alpha, a, lda, b, ldb, c,
                     ldc);
}

// C = diag(x) * A (side = 0) or A * diag(x) (side = 1). Native in cuBLAS.
__attribute__((tessera_op("cublas.dgmm(h, side, m, n, a, lda, x, incx, c, "
                          "ldc)"),
               noinline))
static cublasStatus_t tl_ddgmm(cublasHandle_t h, cublasSideMode_t side, int m,
                               int n, const double *a, int lda,
                               const double *x, int incx, double *c, int ldc) {
  return cublasDdgmm(h, side, m, n, a, lda, x, incx, c, ldc);
}

// --- One O(n^2) helper cuBLAS lacks ----------------------------------------
// tl_symmetrize: mirror the lower triangle of a column-major n x n device
// matrix into its upper triangle (syrk writes one triangle only). The CUDA
// kernel lives in tl_cuda.cu. It returns a status because a rule's last call
// replaces the matched call and must have the same result type (cuBLAS calls
// return cublasStatus_t).
#ifdef __cplusplus
extern "C"
#endif
int tl_symmetrize_dev(int n, double *c, int ldc);

__attribute__((tessera_op("tl.symmetrize(n, c, ldc)"), noinline))
static cublasStatus_t tl_symmetrize(int n, double *c, int ldc) {
  return tl_symmetrize_dev(n, c, ldc) ? CUBLAS_STATUS_EXECUTION_FAILED
                                      : CUBLAS_STATUS_SUCCESS;
}

#ifndef TESSERA_NO_RULES
// --- Rules ------------------------------------------------------------------
// L3. Gram product A^T A or A A^T: GEMM -> SYRK + mirror.   Linnea: SYRK
#pragma optimize "if ta == 1 && tb == 0 && m == n && lda == ldb && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, a, ldb, beta, c, ldc) -> cublas.syrk(h, 0, 1, n, k, alpha, a, lda, beta, c, ldc); tl.symmetrize(n, c, ldc)"
#pragma optimize "if ta == 0 && tb == 1 && m == n && lda == ldb && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, a, ldb, beta, c, ldc) -> cublas.syrk(h, 0, 0, n, k, alpha, a, lda, beta, c, ldc); tl.symmetrize(n, c, ldc)"

// L4. Triangular factor in a product: GEMM -> TRMM.         Linnea: TRMM
// cublasDtrmm writes a separate output, so no copy is needed.
#pragma optimize "if triangular_upper(a) && ta == 0 && tb == 0 && m == k && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> cublas.trmm(h, 0, 1, 0, 0, m, n, alpha, a, lda, b, ldb, c, ldc)"
#pragma optimize "if triangular_lower(a) && ta == 0 && tb == 0 && m == k && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> cublas.trmm(h, 0, 0, 0, 0, m, n, alpha, a, lda, b, ldb, c, ldc)"

// L5. Diagonal factor in a product: GEMM -> DGMM.           Linnea: DIAGSCAL
// cublasDdgmm has no alpha, so the rules require alpha == 1.
#pragma optimize "if diagonal(b) && ta == 0 && tb == 0 && k == n && alpha == 1.0 && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> cublas.dgmm(h, 1, m, n, a, lda, b, ldb + 1, c, ldc)"
#pragma optimize "if diagonal(a) && ta == 0 && tb == 0 && m == k && alpha == 1.0 && beta == 0.0, cublas.gemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> cublas.dgmm(h, 0, m, n, b, ldb, a, lda + 1, c, ldc)"

#endif // TESSERA_NO_RULES

#endif

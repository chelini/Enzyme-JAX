#include "matrices.h"

// Linnea: linnea/algebra/transformations.py, transpose(Transpose(A)) -> A.
// Run with: bash run.sh /path/to/Reactant/enzyme/bazel-bin /tmp/tessera-demo
extern "C" {
[[gnu::noinline, pure_tessera_op("la.transpose(a:val=in, out:val=out)")]]
void transpose(const Matrix<3, 3> *a, Matrix<3, 3> *out) {
  Matrix<3, 3> result;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      result.data[c][r] = a->data[r][c];
  *out = result;
}

#pragma optimize "la.transpose(la.transpose(a)) -> a"

// CHECK-LABEL: llvm.func {{.*}}@double_transpose(
// CHECK-NOT: llvm.call @transpose
// CHECK: llvm.return
[[gnu::noinline]]
void double_transpose(const Matrix<3, 3> *a, Matrix<3, 3> *out) {
  Matrix<3, 3> t;
  transpose(a, &t);
  transpose(&t, out);
}

// The marker promises a pure dense product and explicitly permits changed
// floating-point rounding. Nested arrays expose the static dimensions.
// Every candidate kernel must be available before matrix-chain planning.
// Delay output writes so the kernels have value semantics even with aliases.
#define MATMUL(M, K, N)                                                        \
  [[gnu::noinline, clang::annotate("tessera_reassociate_matmul"),              \
    pure_tessera_op("la.mm_" #M "_" #K "_" #N                                  \
                    "(a:val=in, b:val=in, out:val=out)")]]                     \
  void mm_##M##_##K##_##N(const Matrix<M, K> *a, const Matrix<K, N> *b,        \
                          Matrix<M, N> *out) {                                 \
    Matrix<M, N> result;                                                       \
    for (int r = 0; r < M; ++r)                                                \
      for (int c = 0; c < N; ++c) {                                            \
        double sum = 0;                                                        \
        for (int k = 0; k < K; ++k)                                            \
          sum += a->data[r][k] * b->data[k][c];                                \
        result.data[r][c] = sum;                                               \
      }                                                                        \
    *out = result;                                                             \
  }

MATMUL(20, 2, 30)
MATMUL(20, 30, 3)
MATMUL(2, 30, 3)
MATMUL(20, 2, 3)
#undef MATMUL

// (A B) C costs 3000 scalar multiplications; A (B C) costs 300.
// CHECK-LABEL: llvm.func {{.*}}@matrix_chain(
// CHECK-NOT: llvm.call @mm_20_2_30
// CHECK-NOT: llvm.call @mm_20_30_3
// CHECK: llvm.call @mm_2_30_3
// CHECK-NOT: llvm.call @mm_20_2_30
// CHECK-NOT: llvm.call @mm_20_30_3
// CHECK: llvm.call @mm_20_2_3
// CHECK-NOT: llvm.call @mm_20_2_30
// CHECK-NOT: llvm.call @mm_20_30_3
// CHECK: llvm.return
[[gnu::noinline]]
void matrix_chain(const Matrix<20, 2> *a, const Matrix<2, 30> *b,
                  const Matrix<30, 3> *c, Matrix<20, 3> *out) {
  Matrix<20, 30> ab;
  mm_20_2_30(a, b, &ab);
  mm_20_30_3(&ab, c, out);
}
}

#include "matrices.h"

// Linnea's trick4: A^T B + B^T A + A^T C A, with symmetric C.
// Keep calls visible until Tessera; delay writes to preserve value semantics.
extern "C" {
[[gnu::noinline, pure_tessera_op("su.transpose(a:val=in, out:val=out)")]]
void su_transpose(const Matrix<4, 3> *a, Matrix<3, 4> *out) {
  Matrix<3, 4> result;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      result.data[j][i] = a->data[i][j];
  *out = result;
}

[[gnu::noinline, pure_tessera_op("su.mm(a:val=in, b:val=in, out:val=out)")]]
void su_mm(const Matrix<3, 4> *a, const Matrix<4, 3> *b, Matrix<3, 3> *out) {
  Matrix<3, 3> result{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        result.data[i][j] += a->data[i][k] * b->data[k][j];
  *out = result;
}

[[gnu::noinline, pure_tessera_op("su.ca(c:val=in, a:val=in, out:val=out)")]]
void su_ca(const FlatMatrix4 *c, const Matrix<4, 3> *a, Matrix<4, 3> *out) {
  Matrix<4, 3> result{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        result.data[i][j] += c->data[4 * i + k] * a->data[k][j];
  *out = result;
}

[[gnu::noinline, pure_tessera_op("su.add(a:val=in, b:val=in, out:val=out)")]]
void su_add(const Matrix<3, 3> *a, const Matrix<3, 3> *b, Matrix<3, 3> *out) {
  Matrix<3, 3> result;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      result.data[i][j] = a->data[i][j] + b->data[i][j];
  *out = result;
}

[[gnu::noinline,
  pure_tessera_op("su.axpy(alpha, x:val=in, b:val=in, out:val=out)")]]
void su_axpy(double alpha, const Matrix<4, 3> *x, const Matrix<4, 3> *b,
             Matrix<4, 3> *out) {
  Matrix<4, 3> result;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      result.data[i][j] = b->data[i][j] + alpha * x->data[i][j];
  *out = result;
}

// SYR2K with transposed inputs, alpha=1 and beta=0. Materialize both triangles
// because the surrounding API promises a full matrix, unlike BLAS SYR2K.
[[gnu::noinline, pure_tessera_op("su.syr2k(a:val=in, t:val=in, out:val=out)"),
  tessera_guarantees("symmetric(out)")]]
void su_syr2k(const Matrix<4, 3> *a, const Matrix<4, 3> *t, Matrix<3, 3> *out) {
  Matrix<3, 3> result;
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j) {
      double sum = 0;
      for (int k = 0; k < 4; ++k)
        sum += a->data[k][i] * t->data[k][j] + t->data[k][i] * a->data[k][j];
      result.data[i][j] = result.data[j][i] = sum;
    }
  *out = result;
}

// This user-supplied algebraic rule explicitly permits changed FP rounding.
#pragma optimize                                                               \
    "if symmetric(c), su.add(su.add(su.mm(su.transpose(a), b), su.mm(su.transpose(b), a)), su.mm(su.transpose(a), su.ca(c, a))) -> su.syr2k(a, su.axpy(0.5, su.ca(c, a), b))"

// CHECK-LABEL: llvm.func {{.*}}@symmetric_update_known(
// CHECK-NOT: llvm.cond_br
// CHECK-NOT: llvm.call @su_transpose
// CHECK-NOT: llvm.call @su_mm
// CHECK-NOT: llvm.call @su_add
// CHECK: llvm.call @su_ca
// CHECK: llvm.call @su_axpy
// CHECK-NOT: llvm.call @su_ca
// CHECK-NOT: llvm.call @su_axpy
// CHECK: llvm.call @su_syr2k
// CHECK-NOT: llvm.call
// CHECK: llvm.return
[[gnu::noinline, tessera_assumes("symmetric(c)")]]
void symmetric_update_known(const Matrix<4, 3> *a, const Matrix<4, 3> *b,
                            const FlatMatrix4 *c, Matrix<3, 3> *out) {
  Matrix<3, 4> at, bt;
  Matrix<4, 3> ca;
  Matrix<3, 3> atb, bta, sum, atca;
  su_transpose(a, &at);
  su_transpose(b, &bt);
  su_mm(&at, b, &atb);
  su_mm(&bt, a, &bta);
  su_add(&atb, &bta, &sum);
  su_ca(c, a, &ca);
  su_mm(&at, &ca, &atca);
  su_add(&sum, &atca, out);
}

// CHECK-LABEL: llvm.func {{.*}}@symmetric_update_checked(
// CHECK: llvm.fcmp "oeq"
// CHECK: llvm.cond_br
// CHECK: llvm.call @su_ca
// CHECK: llvm.call @su_axpy
// CHECK: llvm.call @su_syr2k
// CHECK: llvm.br
// CHECK: llvm.call @su_transpose
// CHECK: llvm.call @su_transpose
// CHECK: llvm.call @su_mm
// CHECK: llvm.call @su_mm
// CHECK: llvm.call @su_add
// CHECK: llvm.call @su_ca
// CHECK: llvm.call @su_mm
// CHECK: llvm.call @su_add
// CHECK: llvm.return
[[gnu::noinline]]
void symmetric_update_checked(const Matrix<4, 3> *a, const Matrix<4, 3> *b,
                              const FlatMatrix4 *c, Matrix<3, 3> *out) {
  Matrix<3, 4> at, bt;
  Matrix<4, 3> ca;
  Matrix<3, 3> atb, bta, sum, atca;
  su_transpose(a, &at);
  su_transpose(b, &bt);
  su_mm(&at, b, &atb);
  su_mm(&bt, a, &bta);
  su_add(&atb, &bta, &sum);
  su_ca(c, a, &ca);
  su_mm(&at, &ca, &atca);
  su_add(&sum, &atca, out);
}
}

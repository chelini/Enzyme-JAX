#pragma once

template <int Rows, int Cols> struct Matrix {
  double data[Rows][Cols];
};

extern "C" void double_transpose(const Matrix<3, 3> *, Matrix<3, 3> *);
extern "C" void matrix_chain(const Matrix<20, 2> *, const Matrix<2, 30> *,
                             const Matrix<30, 3> *, Matrix<20, 3> *);

// Flat row-major storage is understood by Tessera's runtime property checker.
struct FlatMatrix4 {
  double data[16];
};
extern "C" void symmetric_update_known(const Matrix<4, 3> *,
                                       const Matrix<4, 3> *,
                                       const FlatMatrix4 *, Matrix<3, 3> *);
extern "C" void symmetric_update_checked(const Matrix<4, 3> *,
                                         const Matrix<4, 3> *,
                                         const FlatMatrix4 *, Matrix<3, 3> *);

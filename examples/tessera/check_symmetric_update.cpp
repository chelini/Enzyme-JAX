#include "matrices.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>

// Evaluate the original expression independently, with wider accumulators.
static void check(const Matrix<4, 3> &a, const Matrix<4, 3> &b,
                  const FlatMatrix4 &c, const Matrix<3, 3> &result) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      long double reference = 0;
      for (int k = 0; k < 4; ++k) {
        reference += static_cast<long double>(a.data[k][i]) * b.data[k][j] +
                     static_cast<long double>(b.data[k][i]) * a.data[k][j];
        for (int l = 0; l < 4; ++l)
          reference += static_cast<long double>(a.data[k][i]) *
                       c.data[4 * k + l] * a.data[l][j];
      }
      assert(std::isfinite(result.data[i][j]));
      assert(std::abs(result.data[i][j] - reference) <
             1e-11L * (1 + std::abs(reference)));
    }
}

int main() {
  int checked = 0;
  for (int seed = 0; seed < 32; ++seed) {
    Matrix<4, 3> a, b;
    FlatMatrix4 c;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 3; ++j) {
        a.data[i][j] = ((i * 7 + j * 3 + seed) % 23 - 11) / 13.0;
        b.data[i][j] = ((i * 5 + j * 11 + seed) % 19 - 9) / 17.0;
      }
    for (int i = 0; i < 4; ++i)
      for (int j = i; j < 4; ++j)
        c.data[4 * i + j] = c.data[4 * j + i] =
            ((i * 3 + j * 7 + seed) % 17 - 8) / 11.0;
    // Opposite signs on the diagonal ensure C is indefinite, not SPD.
    c.data[0] = -2;
    c.data[5] = 3;
    for (bool aliasedInputs : {false, true}) {
      const auto &rhs = aliasedInputs ? a : b;
      Matrix<3, 3> known, dynamic;
      symmetric_update_known(&a, &rhs, &c, &known);
      symmetric_update_checked(&a, &rhs, &c, &dynamic);
      check(a, rhs, c, known);
      check(a, rhs, c, dynamic);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          assert(known.data[i][j] == known.data[j][i]);
          assert(dynamic.data[i][j] == known.data[i][j]);
        }
      checked += 18;
    }

    // Breaking symmetry must take the original path, whose result need not
    // be symmetric. Never call the function promising symmetry with this C.
    c.data[1] += 2;
    Matrix<3, 3> fallback;
    symmetric_update_checked(&a, &b, &c, &fallback);
    check(a, b, c, fallback);
    double asymmetry = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        asymmetry += std::abs(fallback.data[i][j] - fallback.data[j][i]);
    assert(asymmetry > 1e-6);
    checked += 9;
  }
  std::printf("PASS: %d symmetric-update elements (known, guarded, aliased "
              "inputs, nonsymmetric fallback)\n",
              checked);
}

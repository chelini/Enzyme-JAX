#include "matrices.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
  const std::uint64_t bits[] = {0,
                                0x8000000000000000ULL,
                                0x3ff0000000000000ULL,
                                0x7ff0000000000000ULL,
                                0xfff0000000000000ULL,
                                0x7ff8000000000042ULL,
                                0x4008000000000000ULL,
                                0xbfe0000000000000ULL,
                                1};
  Matrix<3, 3> input, transposed;
  static_assert(sizeof(input) == sizeof(bits));
  std::memcpy(&input, bits, sizeof(input));
  double_transpose(&input, &transposed);
  assert(std::memcmp(&input, &transposed, sizeof(input)) == 0);
  double_transpose(&input, &input);
  assert(std::memcmp(&input, bits, sizeof(input)) == 0);

  int checked = 0;
  for (int seed = 0; seed < 32; ++seed) {
    Matrix<20, 2> a;
    Matrix<2, 30> b;
    Matrix<30, 3> c;
    Matrix<20, 3> result;
    for (int i = 0; i < 20; ++i)
      for (int j = 0; j < 2; ++j)
        a.data[i][j] = ((i * 7 + j * 3 + seed) % 23 - 11) / 13.0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 30; ++j)
        b.data[i][j] = ((i * 5 + j * 11 + seed) % 19 - 9) / 17.0;
    for (int i = 0; i < 30; ++i)
      for (int j = 0; j < 3; ++j)
        c.data[i][j] = ((i * 3 + j * 7 + seed) % 17 - 8) / 11.0;
    matrix_chain(&a, &b, &c, &result);
    for (int i = 0; i < 20; ++i)
      for (int j = 0; j < 3; ++j) {
        long double reference = 0;
        for (int k = 0; k < 2; ++k)
          for (int l = 0; l < 30; ++l)
            reference += static_cast<long double>(a.data[i][k]) * b.data[k][l] *
                         c.data[l][j];
        assert(std::isfinite(result.data[i][j]));
        assert(std::abs(result.data[i][j] - reference) <
               1e-11L * (1 + std::abs(reference)));
        ++checked;
      }
  }
  std::printf("PASS: bitwise double transpose and %d matrix-chain elements\n",
              checked);
}

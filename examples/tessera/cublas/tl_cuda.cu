// Device kernel behind tl_symmetrize: c[j + i*ldc] = c[i + j*ldc] for i > j.
#include <cuda_runtime.h>

__global__ void symmetrize_kernel(int n, double *c, int ldc) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // column
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row
  if (i < n && j < i)
    c[j + (size_t)i * ldc] = c[i + (size_t)j * ldc];
}

extern "C" int tl_symmetrize_dev(int n, double *c, int ldc) {
  dim3 b(32, 8), g((n + b.x - 1) / b.x, (n + b.y - 1) / b.y);
  symmetrize_kernel<<<g, b>>>(n, c, ldc);
  return (int)cudaGetLastError();
}

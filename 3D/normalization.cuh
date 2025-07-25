#ifndef NORMALIZATION_CUH
#define NORMALIZATION_CUH

#include <algorithm>
#include <omp.h>


template <typename T>
__global__ void normalize_gpu_data(T* x, T* y, T* z, int n, T min_x, T max_x, T min_y, T max_y, T min_z, T max_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = (x[i] - min_x) / (max_x - min_x);
        y[i] = (y[i] - min_y) / (max_y - min_y);
        z[i] = (z[i] - min_z) / (max_z - min_z);
    }
}

template <typename T>
void normalize_cpu_data(T* x, T* y, T* z, int n) {
    T min_x = *std::min_element(x, x + n);
    T max_x = *std::max_element(x, x + n);
    T min_y = *std::min_element(y, y + n);
    T max_y = *std::max_element(y, y + n);
    T min_z = *std::min_element(z, z + n);
    T max_z = *std::max_element(z, z + n);

    for (int i = 0; i < n; ++i) {
        x[i] = (x[i] - min_x) / (max_x - min_x);
        y[i] = (y[i] - min_y) / (max_y - min_y);
        z[i] = (z[i] - min_z) / (max_z - min_z);
    }
}

template <typename T>
void normalize_cpu_data_omp(T* x, T* y, T* z, int n) {
    T x_min = *std::min_element(x, x + n);
    T x_max = *std::max_element(x, x + n);
    T y_min = *std::min_element(y, y + n);
    T y_max = *std::max_element(y, y + n);
    T z_min = *std::min_element(z, z + n);
    T z_max = *std::max_element(z, z + n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (x_max != x_min) x[i] = (x[i] - x_min) / (x_max - x_min);
        if (y_max != y_min) y[i] = (y[i] - y_min) / (y_max - y_min);
        if (z_max != z_min) z[i] = (z[i] - z_min) / (z_max - z_min);
    }
}

#endif // NORMALIZATION_CUH

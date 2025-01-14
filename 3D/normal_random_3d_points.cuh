#include <omp.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>

template <typename T>
__global__ void kernel_generate_random_normal_points_gpu(T* x, T* y, T* z, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);
        
        // Parameters for the normal distribution
        float mean = 0.5f;
        float stddev = 0.1f;

        x[i] = mean + stddev * curand_normal(&state);
        y[i] = mean + stddev * curand_normal(&state);
        z[i] = mean + stddev * curand_normal(&state);
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

template <typename T>
void generate_random_normal_points_gpu(int n, T* d_x, T* d_y, T* d_z, unsigned long long seed) {
    // Define the number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    kernel_generate_random_normal_points_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, n, seed);

    // Check for errors in kernel launch
    checkCudaError(cudaGetLastError(), "[kernel_generate_random_normal_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
}

template <typename T>
void generate_random_normal_points_omp(int n, T* x, T* y, T* z, unsigned long long seed) {
    double mean = 0.5;
    double stddev = 0.1;

    // Usar una distribución normal con la media y desviación estándar especificadas
    std::normal_distribution<T> dist(mean, stddev);

    #pragma omp parallel  // Iniciar una sección paralela
    {
        std::mt19937_64 generator(seed + omp_get_thread_num());  // Crear un generador de números aleatorios con una semilla única para cada hilo

        #pragma omp for  // Paralelizar el bucle for
        for(int i = 0; i < n; ++i) {
            x[i] = dist(generator);
            y[i] = dist(generator);
            z[i] = dist(generator);
        }
    }
}

template <typename T>
void generate_random_normal_points(int n, T* x, T* y, T* z, unsigned long long seed) {
    double mean = 0.5;
    double stddev = 0.1;

    std::mt19937_64 generator(seed);
    std::normal_distribution<T> dist(mean, stddev);

    for(int i = 0; i < n; ++i) {
        x[i] = dist(generator);
        y[i] = dist(generator);
        z[i] = dist(generator);
    }
}

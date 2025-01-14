#include <random>
#include <curand.h>
#include <curand_kernel.h>

template <typename T>
__global__ void kernel_generate_random_uniform_points_gpu(T* x, T* y, T* z, int n, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);

        // Parameters for the normal distribution
        float mean = 0.5f;
        float stddev = 0.1f;

        x[i] = mean + stddev * curand_uniform(&state);
        y[i] = mean + stddev * curand_uniform(&state);
        z[i] = mean + stddev * curand_uniform(&state);
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

template <typename T>
void generate_random_uniform_points_gpu(int n, T* d_x, T* d_y, T* d_z, unsigned long seed) {
    // Define the number of threads and blocks
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;

    kernel_generate_random_uniform_points_gpu<<<numBlocks, blockSize>>>(d_x, d_y, d_z, n, seed);

    // Check for errors in kernel launch
    checkCudaError(cudaGetLastError(), "[kernel_generate_random_uniform_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
}

#include <omp.h>

template <typename T>
void generate_random_uniform_points_omp(int n, T* x, T* y, T* z, unsigned long seed) {
    std::uniform_real_distribution<T> dist(0.5, 0.1);

    #pragma omp parallel  // Iniciar una sección paralela
    {
        std::mt19937 generator(seed + omp_get_thread_num());  // Crear un generador de números aleatorios por hilo

        #pragma omp for  // Paralelizar el bucle for
        for(int i = 0; i < n; ++i) {
            x[i] = dist(generator);
            y[i] = dist(generator);
            z[i] = dist(generator);
        }
    }
}

template <typename T>
void generate_random_uniform_points(int n, T* x, T* y, T* z, unsigned long seed) {
    std::mt19937 generator(seed);  // Crear un generador de números aleatorios
    std::uniform_real_distribution<T> dist(0.5, 0.1);
    for(int i = 0; i < n; i++) {
        x[i] = dist(generator);
        y[i] = dist(generator);
        z[i] = dist(generator);
    }
}

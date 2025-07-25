#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <set>
#include <tuple>
#include <mutex>

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
    // Definir el generador de números aleatorios y la distribución uniforme
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // Usar OpenMP para paralelizar la generación de números aleatorios
    #pragma omp parallel
    {
        std::mt19937 local_generator(seed + omp_get_thread_num()); // Semilla única para cada hilo
        std::uniform_real_distribution<T> local_distribution(0.0, 1.0);
        
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            x[i] = local_distribution(local_generator);
            y[i] = local_distribution(local_generator);
            z[i] = local_distribution(local_generator);
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

template <typename T>
void generate_unique_random_uniform_points(int n, T* x, T* y, T* z, unsigned long seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> dist(0.5, 0.1);

    std::set<std::tuple<T, T, T>> unique_points;

    int count = 0;
    while (count < n) {
        T new_x = dist(generator);
        T new_y = dist(generator);
        T new_z = dist(generator);

        auto point = std::make_tuple(new_x, new_y, new_z);
        if (unique_points.insert(point).second) {  // Insert only if the point is unique
            x[count] = new_x;
            y[count] = new_y;
            z[count] = new_z;
            count++;
        }
    }
}

template <typename T>
void generate_unique_random_uniform_points_omp(int n, T* x, T* y, T* z, unsigned long seed) {
    std::set<std::tuple<T, T, T>> unique_points;
    std::mutex set_mutex;

    int count = 0;

    #pragma omp parallel
    {
        std::mt19937 generator(seed + omp_get_thread_num());
        std::uniform_real_distribution<T> dist(0.5, 0.1);

        while (true) {
            T new_x = dist(generator);
            T new_y = dist(generator);
            T new_z = dist(generator);

            auto point = std::make_tuple(new_x, new_y, new_z);

            bool inserted = false;
            {
                std::lock_guard<std::mutex> lock(set_mutex);
                if (unique_points.size() < n && unique_points.insert(point).second) {
                    int idx = count++;
                    if (idx < n) {
                        x[idx] = new_x;
                        y[idx] = new_y;
                        z[idx] = new_z;
                        inserted = true;
                    }
                }
            }

            if (count >= n) break;
        }
    }
}

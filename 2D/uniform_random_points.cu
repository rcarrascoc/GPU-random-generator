
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// Define the REAL type as float
#define REAL float
#define REPEAT 10

#include "uniform_random_points.cuh"


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [gpu/omp/seq] [n_points]" << std::endl;
        return 1;
    }
    std::string mode(argv[1]);
    int n = std::stoi(argv[2]);

    REAL *x, *y, *d_x, *d_y;

    if (mode == "gpu") {
        #ifdef USE_GPU
        // Allocate memory on GPU
        checkCudaError(cudaMalloc(&d_x, n * sizeof(REAL)), "cudaMalloc d_x failed");
        checkCudaError(cudaMalloc(&d_y, n * sizeof(REAL)), "cudaMalloc d_y failed");
        #else
        std::cerr << "GPU mode not supported. Compile with -DUSE_GPU flag." << std::endl;
        return 1;
        #endif
        #ifdef SAVE_OFF
        // Allocate memory on CPU
        x = new REAL[n];
        y = new REAL[n];
        #endif
    } else {
        // Allocate memory on CPU
        x = new REAL[n];
        y = new REAL[n];
    }

    checkCudaError(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "gpu") {
        #ifdef USE_GPU
        for (int i = 0; i < REPEAT; i++)
            generate_random_uniform_points_gpu<REAL>(n, d_x, d_y);
        #else
        std::cerr << "GPU mode not supported. Compile with -DUSE_GPU flag." << std::endl;
        return 1;
        #endif
    } else if (mode == "omp") {
        for (int i = 0; i < REPEAT; i++)
            generate_random_uniform_points_omp<REAL>(n, x, y);
    } else if (mode == "seq") {
        for (int i = 0; i < REPEAT; i++)
            generate_random_uniform_points<REAL>(n, x, y);
    } else {
        std::cerr << "Invalid mode. Use 'gpu', 'omp', or 'seq'." << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken (" << mode << "): " << elapsed.count() << " seconds." << std::endl;

    #ifdef SAVE_OFF
    // Write to .off file
    std::string filename = "points_uniform_" + mode + ".off";
    std::ofstream offFile(filename);
    offFile << "OFF\n";
    offFile << n << " 0 0\n";
    if (mode == "gpu") {
        #ifdef USE_GPU
        // Allocate host memory for GPU data
        x = new REAL[n];
        y = new REAL[n];
        // Copy from device to host
        checkCudaError(cudaMemcpy(x, d_x, n * sizeof(REAL), cudaMemcpyDeviceToHost), "cudaMemcpy d_x to x failed");
        checkCudaError(cudaMemcpy(y, d_y, n * sizeof(REAL), cudaMemcpyDeviceToHost), "cudaMemcpy d_y to y failed");
        #endif
    }
    for (int i = 0; i < n; i++) {
        offFile << x[i] << " " << y[i] << " 0\n";
    }
    offFile.close();
    #endif

    if (mode == "gpu") {
        #ifdef USE_GPU
        // Free memory on GPU
        cudaFree(d_x);
        cudaFree(d_y);
        #endif        
        #ifdef SAVE_OFF
        // Free the host memory allocated for GPU data
        delete[] x;
        delete[] y;
        #endif
    } else {
        // Free allocated memory on CPU
        delete[] x;
        delete[] y;
    }

    return 0;
}

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// Define the REAL type as float
#define REAL float

#define REPEAT 1

#include "sphere_random_3d_points.cuh"
#include "normalization.cuh"

int main(int argc, char* argv[]) {
    
    
    if (argc != 
        #ifdef SAVE_OFF 
        6
        #else
        5
        #endif
    ) {
        std::cerr << "Usage: " << argv[0] << " [gpu/omp/seq] [n_points] [prob] [seed]"
        #ifdef SAVE_OFF
        "[output_name]"
        #endif
        << std::endl;
        return 1;
    }

    std::string mode(argv[1]);
    int n = std::stoi(argv[2]);
    double prob = std::stod(argv[3]);
    unsigned long seed = std::stoul(argv[4]);
    #ifdef SAVE_OFF
    std::string output_name(argv[5]);
    #endif


    REAL *x, *y, *z, *d_x, *d_y, *d_z;

    if (mode == "gpu") {
        #ifdef USE_GPU
        // Allocate memory on GPU
        checkCudaError(cudaMalloc(&d_x, n * sizeof(REAL)), "cudaMalloc d_x failed");
        checkCudaError(cudaMalloc(&d_y, n * sizeof(REAL)), "cudaMalloc d_y failed");
        checkCudaError(cudaMalloc(&d_z, n * sizeof(REAL)), "cudaMalloc d_z failed");
        #else
        std::cerr << "GPU mode not supported. Compile with -DUSE_GPU flag." << std::endl;
        return 1;
        #endif
        #ifdef SAVE_OFF
        // Allocate memory on CPU
        x = new REAL[n];
        y = new REAL[n];
        z = new REAL[n];
        #endif
    } else {
        // Allocate memory on CPU
        x = new REAL[n];
        y = new REAL[n];
        z = new REAL[n];
    }

    checkCudaError(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "gpu") {
        #ifdef USE_GPU
        for (int i = 0; i < REPEAT; i++)
            generate_random_sphere_points_gpu<REAL>(n, d_x, d_y, d_z, prob, seed);

        #else
        std::cerr << "GPU mode not supported. Compile with -DUSE_GPU flag." << std::endl;
        return 1;
        #endif
    } else if (mode == "omp") {
        for (int i = 0; i < REPEAT; i++)
            generate_random_sphere_points_omp<REAL>(n, x, y, z, prob, seed);

        // Normalize CPU data in parallel
        normalize_cpu_data_omp(x, y, z, n);
    } else if (mode == "seq") {
        for (int i = 0; i < REPEAT; i++)
            generate_random_sphere_points<REAL>(n, x, y, z, prob, seed);

        // Normalize CPU data
        normalize_cpu_data(x, y, z, n);
    } else {
        std::cerr << "Invalid mode. Use 'gpu', 'omp', or 'seq'." << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken (" << mode << "): " << elapsed.count()/REPEAT << " seconds." << std::endl;

    #ifdef SAVE_OFF
    // Write to .off file
    std::string filename = output_name + ".off";
    std::ofstream offFile(filename);
    offFile << "OFF\n";
    offFile << n << " 0 0\n";
    if (mode == "gpu") {
        #ifdef USE_GPU
        // Allocate host memory for GPU data
        x = new REAL[n];
        y = new REAL[n];
        z = new REAL[n];
        // Copy from device to host
        checkCudaError(cudaMemcpy(x, d_x, n * sizeof(REAL), cudaMemcpyDeviceToHost), "cudaMemcpy d_x to x failed");
        checkCudaError(cudaMemcpy(y, d_y, n * sizeof(REAL), cudaMemcpyDeviceToHost), "cudaMemcpy d_y to y failed");
        checkCudaError(cudaMemcpy(z, d_z, n * sizeof(REAL), cudaMemcpyDeviceToHost), "cudaMemcpy d_z to z failed");
        #endif
    }
    for (int i = 0; i < n; i++) {
        offFile << x[i] << " " << y[i] << " " << z[i] << "\n";
    }
    offFile.close();
    #endif

    if (mode == "gpu") {
        #ifdef USE_GPU
        // Free memory on GPU
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        #endif        
        #ifdef SAVE_OFF
        // Free the host memory allocated for GPU data
        delete[] x;
        delete[] y;
        delete[] z;
        #endif
    } else {
        // Free allocated memory on CPU
        delete[] x;
        delete[] y;
        delete[] z;
    }

    return 0;
}

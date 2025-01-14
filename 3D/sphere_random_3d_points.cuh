#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <omp.h>
#include <cstdlib>

template <typename T>
__device__ void kernel_calculate_point(float alpha, float beta, float ra, T &x, T &y, T &z) {
    float rad_alpha = alpha * M_PI / 180.0;
    float rad_beta = beta * M_PI / 180.0;
    x = ra * sin(rad_alpha) * cos(rad_beta);
    y = ra * sin(rad_alpha) * sin(rad_beta);
    z = ra * cos(rad_alpha);
}

template <typename T>
__global__ void kernel_generate_random_sphere_points(T *X, T *Y, T *Z, int n, double prob, float cx, float cy, float cz, float ra, int N, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);
        
        int aux;    
        float x, y, z, alpha, beta, fact;
        fact = curand_uniform(&state);
        alpha = fact * 180.0;
        beta = curand_uniform(&state) * 360.0;

        if (i < N) {
            kernel_calculate_point(alpha, beta, ra, x, y, z);
        } else {
            float dt = ra * prob;
            fact = curand_uniform(&state);
            aux = curand(&state);
            kernel_calculate_point(alpha, beta, (aux % 2 == 0) ? ra + fact * dt : ra - fact * dt, x, y, z);
        }

        X[i] = cx + x;
        Y[i] = cy + y;
        Z[i] = cz + z;
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

template <typename T>
void generate_random_sphere_points_gpu(int n, T *d_x, T *d_y, T *d_z, double prob, unsigned long seed) {
    // Use current time as a seed for the random number generator
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int N = n - static_cast<int>(n * prob);
    kernel_generate_random_sphere_points<<<numBlocks, blockSize>>>(d_x, d_y, d_z, n, prob, 0.5f, 0.5f, 0.5f, 1.0f, N, seed);

    // Check for errors in kernel launch
    checkCudaError(cudaGetLastError(), "[kernel_generate_random_uniform_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
} 


#include <cmath>
#include <random>

template <typename T>
void calculate_point(float alpha, float beta, float ra, T &x, T &y, T &z) {
    float rad_alpha = alpha * M_PI / 180.0;
    float rad_beta = beta * M_PI / 180.0;
    x = ra * sin(rad_alpha) * cos(rad_beta);
    y = ra * sin(rad_alpha) * sin(rad_beta);
    z = ra * cos(rad_alpha);
}

template <typename T>
void generate_random_sphere_points_omp(int n, T *X, T *Y, T *Z, double prob, unsigned long seed) {
    srand(seed);
    int N = n - static_cast<int>(n * prob);
    float cx = 0.5, cy = 0.5, cz = 0.5, ra = 1.0;

    #pragma omp parallel
    {
        std::mt19937 gen(rand());

        #pragma omp for
        for (int i = 0; i < n; i++) {
            float x, y, z, alpha, beta, fact;
            fact = static_cast<float>(gen()) / gen.max();
            alpha = fact * 180.0;
            beta = static_cast<float>(gen()) / gen.max() * 360.0;

            if (i < N) {
                calculate_point(alpha, beta, ra, x, y, z);
            } else {
                float dt = ra * prob;
                fact = static_cast<float>(gen()) / gen.max();
                calculate_point(alpha, beta, (gen() % 2 == 0) ? ra + fact * dt : ra - fact * dt, x, y, z);
            }

            X[i] = cx + x;
            Y[i] = cy + y;
            Z[i] = cz + z;
        }
    }
}

template <typename T>
void generate_random_sphere_points(int n, T *X, T *Y, T *Z, double prob, unsigned long seed) {
    srand(seed);
	int i;
	int N = n - n*prob;		// points on the sphere.	CASO 1
	float cx, cy, cz, ra;
	float x, y, z, alpha, beta, fact;
	
	ra = 1.0;
	cx = cy = cz = 0.5;	// Center (0.5, 0.5, 0.5)

	// we generate N random points ON the sphere...
	//srand (time(NULL));
	// FOR PARA EL CASO 1
	for(i=0; i<N; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*180.0;	// generate a random angle: 0 <= alpha <= 180
        beta = static_cast <float> (rand())/static_cast <float> (RAND_MAX) * 360.0;
        calculate_point(alpha, beta, ra, x, y, z);
		//cout << alpha << "(" << x << "," << y << "," << z << ") " << endl;
		// here know x^2 + y^2 + z^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
		Z[i]= cz+z;
	}

	float dt = ra*prob;
	// we generate n-N random points INSIDE OF / OR ARAOUND OF the sphere...
	for(; i<n; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*180.0;	// generate a random angle: 0 <= alpha <= 180
        beta = static_cast <float> (rand())/static_cast <float> (RAND_MAX) * 360.0;
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
        calculate_point(alpha, beta, (rand()%2) ? ra+fact*dt : ra-fact*dt, x, y, z);
		//cout << alpha << "(" << x << "," << y << "," << z << ") " << endl;
		// at this moment: x^2 + y^2 + z^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
		Z[i]= cz+z;
	}
}

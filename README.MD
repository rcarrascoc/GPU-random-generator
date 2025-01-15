This repository contains many methods to generate three types of distributions on GPU, OMP and sequential CPU:
1.- Normal distribution
2.- Uniform distribution
3.- Sphere and displaced sphere distribution

compile as:
    make

or
    make alt

*The "make" and "make alt" commands compile all methods and distributions, however, the make command compile only the benchmark and the "make alt" command compiles all files and saves the results in a .OFF file

run the benchmarks as:
    ./sphere_random_3d_points [gpu/omp/seq] [n_points] [prob] 
    ./normal_random_3d_points [gpu/omp/seq] [n_points]
    ./uniform_random_3d_points [gpu/omp/seq] [n_points]

where [n_point] is an integer number between 1 to $((2**30)), and [prob] can be any float number between 0 to 1.

run and save .OFF file as:
    ./sphere_random_3d_points [gpu/omp/seq] [n_points] [prob] [output_name]
    ./normal_random_3d_points [gpu/omp/seq] [n_points] [output_name]
    ./uniform_random_3d_points [gpu/omp/seq] [n_points] [output_name]

*You can check the distributions using the jupyter file in each directory

Variables:
REPEAT: number of repetitions of the experiment
REAL: type of input and output
USE_GPU: compile with GPU or not
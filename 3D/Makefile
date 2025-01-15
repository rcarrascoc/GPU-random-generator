# This is the makefile for the 3D random points generator
# The makefile has the following options:
# all: compiles all the files
# normal: compiles the normal_random_3d_points.cu file
# normal_off: compiles the normal_random_3d_points.cu file with the save off option
# uniform: compiles the uniform_random_3d_points.cu file
# uniform_off: compiles the uniform_random_3d_points.cu file with the save off option
# sphere: compiles the sphere_random_3d_points.cu file
# sphere_off: compiles the sphere_random_3d_points.cu file with the save off option
# clean: removes the executables
# alt: compiles all the files with the save off option

all: clean normal uniform sphere

alt: clean normal_random_3d_points_saveoff uniform_random_3d_points_saveoff sphere_random_3d_points_saveoff

normal: normal_random_3d_points

normal_off: normal_random_3d_points_saveoff

# this is the command to compile the normal_random_3d_points.cu file
normal_random_3d_points: normal_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_3d_points -DUSE_GPU normal_random_3d_points.cu -lcurand -O3

# this is the command to compile the normal_random_3d_points.cu file with the save off option
normal_random_3d_points_saveoff: normal_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_3d_points -DUSE_GPU -DSAVE_OFF normal_random_3d_points.cu -lcurand -O3

uniform: uniform_random_3d_points

uniform_off: uniform_random_3d_points_saveoff

# this is the command to compile the uniform_random_3d_points.cu file
uniform_random_3d_points: uniform_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_3d_points -DUSE_GPU uniform_random_3d_points.cu -lcurand -O3

# this is the command to compile the uniform_random_3d_points.cu file with the save off option
uniform_random_3d_points_saveoff: uniform_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_3d_points -DUSE_GPU -DSAVE_OFF uniform_random_3d_points.cu -lcurand -O3

sphere: sphere_random_3d_points

sphere_off: sphere_random_3d_points_saveoff

# this is the command to compile the sphere_random_3d_points.cu file
sphere_random_3d_points: sphere_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o sphere_random_3d_points -DUSE_GPU sphere_random_3d_points.cu -lcurand -O3

# this is the command to compile the sphere_random_3d_points.cu file with the save off option
sphere_random_3d_points_saveoff: sphere_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o sphere_random_3d_points -DUSE_GPU -DSAVE_OFF sphere_random_3d_points.cu -lcurand -O3

# this is the command to compile the normal_random_3d_points.cu file
clean:
	rm -f normal_random_3d_points uniform_random_3d_points sphere_random_3d_points

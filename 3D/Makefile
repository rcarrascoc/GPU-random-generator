
all: clean normal_random_3d_points uniform_random_3d_points sphere_random_3d_points

alt: clean normal_random_3d_points_saveoff uniform_random_3d_points_saveoff sphere_random_3d_points_saveoff

normal: normal_random_3d_points

normal_off: normal_random_3d_points_saveoff

normal_random_3d_points: normal_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_3d_points -DUSE_GPU normal_random_3d_points.cu -lcurand -O3

normal_random_3d_points_saveoff: normal_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_3d_points -DUSE_GPU -DSAVE_OFF normal_random_3d_points.cu -lcurand -O3

uniform: uniform_random_3d_points

uniform_off: uniform_random_3d_points_saveoff

uniform_random_3d_points: uniform_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_3d_points -DUSE_GPU uniform_random_3d_points.cu -lcurand -O3

uniform_random_3d_points_saveoff: uniform_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_3d_points -DUSE_GPU -DSAVE_OFF uniform_random_3d_points.cu -lcurand -O3

sphere: sphere_random_3d_points

sphere_off: sphere_random_3d_points_saveoff

sphere_random_3d_points: sphere_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o sphere_random_3d_points -DUSE_GPU sphere_random_3d_points.cu -lcurand -O3

sphere_random_3d_points_saveoff: sphere_random_3d_points.cu
	nvcc -Xcompiler -fopenmp -o sphere_random_3d_points -DUSE_GPU -DSAVE_OFF sphere_random_3d_points.cu -lcurand -O3

clean:
	rm -f normal_random_3d_points uniform_random_3d_points

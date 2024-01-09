
all: clean normal_random_points uniform_random_points circumference_random_points

alt: clean normal_random_points_saveoff uniform_random_points_saveoff

normal: normal_random_points

normal_off: normal_random_points_saveoff

normal_random_points: normal_random_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_points -DUSE_GPU normal_random_points.cu -lcurand -O3

normal_random_points_saveoff: normal_random_points.cu
	nvcc -Xcompiler -fopenmp -o normal_random_points -DUSE_GPU -DSAVE_OFF normal_random_points.cu -lcurand -O3

uniform: uniform_random_points

uniform_off: uniform_random_points_saveoff

uniform_random_points: uniform_random_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_points -DUSE_GPU uniform_random_points.cu -lcurand -O3

uniform_random_points_saveoff: uniform_random_points.cu
	nvcc -Xcompiler -fopenmp -o uniform_random_points -DUSE_GPU -DSAVE_OFF uniform_random_points.cu -lcurand -O3

circumference: circumference_random_points

circumference_off: circumference_random_points_saveoff

circumference_random_points: circumference_random_points.cu
	nvcc -Xcompiler -fopenmp -o circumference_random_points -DUSE_GPU circumference_random_points.cu -lcurand -O3

circumference_random_points_saveoff: circumference_random_points.cu
	nvcc -Xcompiler -fopenmp -o circumference_random_points -DUSE_GPU -DSAVE_OFF circumference_random_points.cu -lcurand -O3

clean:
	rm -f normal_random_points uniform_random_points circumference_random_points

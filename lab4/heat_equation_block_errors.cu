extern "C" {
	#include "heat_equation.h"
}

#include <cstddef>

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>

#define CHECK_CUDA(val) check((val), #val, __LINE__)

static const int BLOCK_SIZE = 8;

void check(cudaError_t err, const char* func, int line) {
	if (err != cudaSuccess) {
		printf("CUDA Runtime Error at line %d\n", line);
		printf("%s %s\n", cudaGetErrorString(err), func);
		exit(err);
	}
}

extern "C" void init_grid(double* grid, unsigned int N) {
	double left_lower = GRID_CORNER_VALUES[0];
	double left_upper = GRID_CORNER_VALUES[1];
	double right_upper = GRID_CORNER_VALUES[2];
	double right_lower = GRID_CORNER_VALUES[3];  

	grid[0] = left_lower; // grid[0][0]
	grid[N - 1] = left_upper; // grid[0][N - 1]
	grid[N * N - 1] = right_upper; //  grid[N - 1][N - 1]
	grid[(N - 1) * N] = right_lower; // grid[N - 1][0]

	// btw all steps are equal cuz of our corner values
	double step_left = fabs((left_upper - left_lower)) / ((double) (N - 1));
	double step_right = fabs((right_upper - right_lower)) / ((double) (N - 1));
	double step_lower = fabs((right_lower - left_lower)) / ((double) (N - 1));
	double step_upper = fabs((right_upper - left_upper)) / ((double) (N - 1));

	// linear interpolation of initial and boundary conditions of equation
	for (int i = 0; i < N; i++) {
		grid[i] = i * step_left; // grid[0, i]
		grid[(N - 1) * N + i] = i * step_right; // grid[N - 1][i] 
		grid[i * N] = i * step_lower; // grid[i][0]
		// for convinience also fill upper border, although it's not truly initial or boundary condition
		grid[i * N + N - 1] = i * step_upper; // grid[i][N - 1]
	}
}

void calc_error(
	double* __restrict__  error, double* __restrict__ block_errors, 
	void** __restrict__  error_temp_storage, unsigned int num_blocks, cudaStream_t stream
) {
	static size_t temp_storage_bytes = 0;
	if (*error_temp_storage == NULL) { // for the first call
		// Determine temporary device storage requirements
		CHECK_CUDA(cub::DeviceReduce::Max(*error_temp_storage, temp_storage_bytes, block_errors, error, num_blocks, stream));
		// Allocate temporary storage
		cudaMalloc(error_temp_storage, temp_storage_bytes);
	}
	CHECK_CUDA(cub::DeviceReduce::Max(*error_temp_storage, temp_storage_bytes, block_errors, error, num_blocks, stream));
}

__device__ unsigned int calc_new_grid_element(
	double* __restrict__ grid1, double* __restrict__ grid2, 
	unsigned int i, unsigned int j, size_t pitch_width
) {
	unsigned int grid_index = j * pitch_width + i;
	grid2[grid_index] = HEAT_COEF * (
		grid1[grid_index - pitch_width] + // A[i - 1][j]
		grid1[grid_index + pitch_width] + // A[i + 1][j]
		grid1[grid_index - 1] + // A[i][j - 1]
		grid1[grid_index + 1]   // A[i][j + 1]
	);
	return grid_index;
}

__global__ void solve_step_with_block_errors(
	double* __restrict__ grid1, double* __restrict__ grid2, 
	double* __restrict__ block_errors, unsigned int N, size_t pitch_width
) {
	// Specialize BlockReduce for a 2D block of type double
	typedef cub::BlockReduce<double, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_SIZE> BlockReduce;
	// Allocate shared memory for BlockReduce
	__shared__ typename BlockReduce::TempStorage temp_storage;

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	double thread_error = 0.0;
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		unsigned int grid_index = calc_new_grid_element(grid1, grid2, i, j, pitch_width);
		thread_error = fabs(grid1[grid_index] - grid2[grid_index]);
	}

	double block_max_error = BlockReduce(temp_storage).Reduce(thread_error, cub::Max{});

	if (threadIdx.x == 0) {
		block_errors[blockIdx.y * gridDim.x + blockIdx.x] = block_max_error;
	}
}

__global__ void solve_step(
	double* __restrict__ grid1, double* __restrict__ grid2, 
	unsigned int N, size_t pitch_width
) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		calc_new_grid_element(grid1, grid2, i, j, pitch_width);
	}
}

extern "C" SOLVE_RESULT solve_heat_equation(
	double* init_grid, unsigned int N, unsigned int max_iter, 
	double error_rate, unsigned int error_calc_interval
) {
	// due to avoid swap optimization we do only even number of iterations
	if (error_calc_interval % 2 == 1) {
		error_calc_interval++;
	}

	unsigned int N_bytes = N * sizeof(double);
	
	// register already present data on host for use in CUDA
	CHECK_CUDA(cudaHostRegister(init_grid, N_bytes * N, cudaHostRegisterDefault));

	double* grid_buffer;
	size_t pitch;
	// efficient allocation of 2D array using pitcht
	CHECK_CUDA(cudaMallocPitch(&grid_buffer, &pitch, N_bytes, N * 2));
	size_t pitch_width = pitch / sizeof(double); // pitch is size in bytes, for indexing we need num elements in pitch
	double* grid1 = grid_buffer;
	double* grid2 = grid_buffer + N * pitch_width;

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// copy data from init_grid asynchronously in different streams
	CHECK_CUDA(cudaMemcpy2DAsync(grid1, pitch, init_grid, N_bytes, N_bytes, N, cudaMemcpyDefault, stream1));
	CHECK_CUDA(cudaMemcpy2DAsync(grid2, pitch, init_grid, N_bytes, N_bytes, N, cudaMemcpyDefault, stream2));
	CHECK_CUDA(cudaDeviceSynchronize());

	dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE); // size of block = BLOCK_SIZE^2 threads
	dim3 num_blocks(ceil((double) N / BLOCK_SIZE), ceil((double) N / BLOCK_SIZE)); // calc number of blocks by grid size
	
	double* error;
	cudaHostAlloc(&error, sizeof(double), cudaHostAllocDefault);

	double* block_errors;
	unsigned int num_errors = num_blocks.x * num_blocks.y;
	cudaMalloc(&block_errors, num_errors * sizeof(double));
	//printf("Number of blocks: %d\n", num_errors);

	void* error_temp_storage = NULL;
	*error = INFINITY;
	int num_iter;
	for (num_iter = 0; num_iter < max_iter && *error > error_rate; num_iter += 2) {
		unsigned int next_num_iter = num_iter + 2;
		// calc error only on every n'th iteration or on last iteration
		if (next_num_iter % error_calc_interval == 0 || next_num_iter >= max_iter) {
			solve_step<<<num_blocks, threads_per_block, 0, stream1>>>(grid1, grid2, N, pitch_width);
			solve_step_with_block_errors<<<num_blocks, threads_per_block, 0, stream1>>>(grid2, grid1, block_errors, N, pitch_width);
			calc_error(error, block_errors, &error_temp_storage, num_errors, stream1);
			CHECK_CUDA(cudaStreamSynchronize(stream1));
		} else {
			solve_step<<<num_blocks, threads_per_block, 0, stream1>>>(grid1, grid2, N, pitch_width);
			solve_step<<<num_blocks, threads_per_block, 0, stream1>>>(grid2, grid1, N, pitch_width);
		}
	}
	double result_error = *error;
	// copy result matrix back to initial grid
	CHECK_CUDA(cudaMemcpy2DAsync(init_grid, N_bytes, grid1, pitch, N_bytes, N, cudaMemcpyDefault));

	cudaHostUnregister(init_grid);
	cudaFreeHost(error);
	cudaFree(grid_buffer);
	cudaFree(block_errors);
	cudaFree(error_temp_storage);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	return (SOLVE_RESULT) {result_error, num_iter};
}

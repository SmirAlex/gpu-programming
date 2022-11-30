#include <cublas_v2.h>
#include "heat_equation.h"

void init_grid(double* grid, int grid_size) {
	double left_lower = GRID_CORNER_VALUES[0];
	double left_upper = GRID_CORNER_VALUES[1];
	double right_upper = GRID_CORNER_VALUES[2];
	double right_lower = GRID_CORNER_VALUES[3];  

	int N = grid_size;
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

#pragma acc routine seq
static int calc_new_grid_element(double* grid1, double* grid2, int N, int i , int j) {
	int grid_index = i * N + j;
	grid2[grid_index] = HEAT_COEF * (
		grid1[grid_index - N] + // A[i - 1][j]
		grid1[grid_index + N] + // A[i + 1][j]
		grid1[grid_index - 1] + // A[i][j - 1]
		grid1[grid_index + 1]   // A[i][j + 1]
	); 
	return grid_index;
}

SOLVE_RESULT solve_heat_equation(double* init_grid, int grid_size, int max_iter, double error_rate, int error_calc_interval) {
	int N = grid_size;
	int N_sqr = N * N;
	double* grid1 = (double*) malloc(sizeof(double) * N_sqr);
	double* grid2 = (double*) malloc(sizeof(double) * N_sqr);
	double error = INFINITY;
	int num_iter = 0;

	// due to avoid swap optimization we do only even number of iterations
	if (error_calc_interval % 2 == 1) {
		error_calc_interval++;
	}

	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		free(grid1);
		free(grid2);
        printf ("CUBLAS initialization failed with error code %d\n", stat);
        return (SOLVE_RESULT) {INFINITY, -1};
    }

	#pragma acc data copy(init_grid [0:N_sqr]) create(grid1 [0:N_sqr], grid2 [0:N_sqr])
	{
		//#pragma acc data present(init_grid[0:N_sqr], grid1 [0:N_sqr], grid2 [0:N_sqr])
		#pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int grid_index = i * N + j;
				grid1[grid_index] = init_grid[grid_index];
				grid2[grid_index] = init_grid[grid_index];
			}
		}

		int failed = 0; // return is not allowed in the block that performs computation on GPU
		for (num_iter = 0; !failed && num_iter < max_iter && error > error_rate; num_iter += 2) {
			#pragma acc data present(grid1 [0:N_sqr], grid2 [0:N_sqr]) 
			#pragma acc kernels async 
			{
				#pragma acc loop independent collapse(2)
				for (int i = 1; i < N - 1; i++) {
					for (int j = 1; j < N - 1; j++) {
						calc_new_grid_element(grid1, grid2, N, i, j);
					}
				}
				#pragma acc loop independent collapse(2)
				for (int i = 1; i < N - 1; i++) {
					for (int j = 1; j < N - 1; j++) {
						calc_new_grid_element(grid2, grid1, N, i, j);
					}
				}
			}
			int next_num_iter = num_iter + 2;
			// calc error only on every n'th iteration or on last iteration
			if (next_num_iter % error_calc_interval == 0 || next_num_iter >= max_iter) {
				#pragma acc wait				
				#pragma acc data present(init_grid [0:N_sqr], grid1 [0:N_sqr], grid2 [0:N_sqr]) copyout(error)
				#pragma acc host_data use_device(init_grid, grid1, grid2, error)
				{
					double* tmp_grid = init_grid;
					int error_index = 0;	
					double alpha = -1.0;

					stat = cublasDcopy(handle, N_sqr, grid1, 1, tmp_grid, 1); // tmp_grid = grid1
					if (stat != CUBLAS_STATUS_SUCCESS) {
						printf ("CUBLAS grid copy failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasDaxpy(handle, N_sqr, &alpha, grid2, 1, tmp_grid, 1);	// tmp_grid += -(grid2)
					if (stat != CUBLAS_STATUS_SUCCESS) {
						printf ("CUBLAS daxpy failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasIdamax(handle, N_sqr, tmp_grid, 1, &error_index); // index with max abs value		
					if (stat != CUBLAS_STATUS_SUCCESS) {
						printf ("CUBLAS idamax failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasDcopy(handle, 1, tmp_grid + error_index, 1, &error, 1);
					if (stat != CUBLAS_STATUS_SUCCESS) {
						printf ("CUBLAS error copy failed with error code %d\n", stat);
						failed = 1;
					}
				}
				error = fabs(error); // although max was calculated by abs value, error here can still be negative
			}	
		}

		#pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int grid_index = i * N + j;
				init_grid[grid_index] = grid1[grid_index];
			}
		}
	}

	cublasDestroy(handle);
	free(grid1);
	free(grid2);

	return (SOLVE_RESULT) {error, num_iter};
}

#include "heat_equation.h"

// no such function in math.h ...
double fmax(double v1, double v2) {
	return v1 > v2 ? v1 : v2;
}

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

SOLVE_RESULT solve_heat_equation(double* grid, int grid_size, int max_iter, double error_rate) {
	int N = grid_size;
	double* current_grid = (double*) malloc(sizeof(double) * N * N);
	double* next_grid = (double*) malloc(sizeof(double) * N * N);
	double error = INFINITY;
	int num_iter = 0;

	#pragma acc data copy(grid [0:N*N]) create(current_grid [0:N*N]) create(next_grid [0:N*N])
	{
		#pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int grid_index = i * N + j;
				current_grid[grid_index] = grid[grid_index];
				next_grid[grid_index] = grid[grid_index];
			}
		}

		for (num_iter = 0; num_iter < max_iter && error > error_rate; num_iter++) {
			error = 0.0;

			#pragma acc data present(next_grid [0:N*N], current_grid [0:N*N]) 
			{
				#pragma acc kernels loop independent collapse(2) reduction(max:error)
				for (int i = 1; i < N - 1; i++) {
					for (int j = 1; j < N - 1; j++) {
						int grid_index = i * N + j;
						next_grid[grid_index] = HEAT_COEF * (
							current_grid[grid_index - N] + // A[i - 1][j]
							current_grid[grid_index + N] + // A[i + 1][j]
							current_grid[grid_index - 1] + // A[i][j - 1]
							current_grid[grid_index + 1]   // A[i][j + 1]
						); 
						error = fmax(error, fabs(next_grid[grid_index] - current_grid[grid_index]));
					}
				}
			}
			double* tmp_grid = current_grid;
			current_grid = next_grid;
			next_grid = tmp_grid;
		}

		#pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int grid_index = i * N + j;
				grid[grid_index] = current_grid[grid_index];
			}
		}
	}

	free(current_grid);
	free(next_grid);

	return (SOLVE_RESULT) {error, num_iter};
}

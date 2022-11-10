#include <time.h>

#include "heat_equation.h"

static const int MEASURE_ATTEMPTS = 5;

double get_time_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec)) * 1000;
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, USAGE);
        return 1;
    }

    double error_rate = strtod(argv[1], NULL);
    if (error_rate < MAX_ERROR_RATE) {
        fprintf(stderr, "Max error rate is %f.\n", MAX_ERROR_RATE);
        return 1;
    }

    int grid_size = atoi(argv[2]);
    if (grid_size != GRID_SIZES[0] && grid_size != GRID_SIZES[1] && grid_size != GRID_SIZES[2]) {
        fprintf(stderr, "Possible grid sizes are only %d^2, %d^2, %d^2.\n", GRID_SIZES[0], GRID_SIZES[1], GRID_SIZES[2]);
        return 1;
    }
  
    int num_iterations = atoi(argv[3]);
    if (num_iterations > MAX_NUM_ITER) {
        fprintf(stderr, "Max number of iterations is %d.\n", MAX_NUM_ITER);
        return 1;
    }

    int error_calc_interval = 1;
    if (argc >= 5) {
        error_calc_interval = atoi(argv[4]);
    }

    struct timespec start, end;
    double total_elapsed = 0;
    SOLVE_RESULT result;
    for (int i = 0; i < MEASURE_ATTEMPTS; i++) {
        double* grid = (double*) calloc((grid_size * grid_size), sizeof(double)); // fill with zeros
        init_grid(grid, grid_size);

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        result = solve_heat_equation(grid, grid_size, num_iterations, error_rate, error_calc_interval);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        total_elapsed += get_time_ms(start, end);
        free(grid);
    }
    double elapsed_avg = total_elapsed / MEASURE_ATTEMPTS;

    printf("Heat equation solved with error %f after %d iterations with average time %f ms\n", 
        result.error, result.num_iterations, elapsed_avg);
    return 0;
}

#ifndef HEAT_EQUATION_H
#define HEAT_EQUATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct  {
    double error;
    int num_iterations;
} SOLVE_RESULT;

static const char USAGE[] = "Usage: ./heat_equation.out [error rate] [grid size] [number of iterations]\n";
static const double MAX_ERROR_RATE = 0.000001;  // 10^-6
static const int MAX_NUM_ITER = 1000000;  // 10^6
static const int GRID_SIZES[] = {128, 256, 512};

static const double GRID_CORNER_VALUES[] = {10, 20, 30, 20};
static const double HEAT_COEF = 0.25;

void init_grid(double* grid, int grid_size);

SOLVE_RESULT solve_heat_equation(double* grid, int grid_size, int max_iter, double error_rate);

#endif //  HEAT_EQUATION_H

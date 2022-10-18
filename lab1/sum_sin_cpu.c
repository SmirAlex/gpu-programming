#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FLP_TYPE double

const int N = pow(10, 7);
const int NUM_ITERATIONS = 5;

 double get_time_ms(struct timespec start, struct timespec end) {
	return (end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec)) * 1000;
 }

FLP_TYPE compute_sum_sin_cpu() {
    FLP_TYPE* sin_arr = (FLP_TYPE*) malloc(sizeof(FLP_TYPE) * N);
    FLP_TYPE step = 2 * M_PI / (FLP_TYPE) (N - 1);
    for (int i = 0; i < N; i++) {
        sin_arr[i] = sin(i * step);
    }
    
    // printf("Created sin array [0; 2PI] with step =  %f.\n", step);
    
    FLP_TYPE sum_sin = 0;
    for (int i = 0; i < N; i++) {
        sum_sin += sin_arr[i];
    }
    free(sin_arr);
    // printf("Sum of sins (integral) = %f.\n", sum_sin);
    return sum_sin;
}

void main() {
    struct timespec start, end;
    double total_elapsed = 0;
    FLP_TYPE sum_sim = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        sum_sim = compute_sum_sin_cpu();
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        total_elapsed += get_time_ms(start, end);
    }
    double elapsed_avg = total_elapsed / NUM_ITERATIONS;
    printf("Sum of sins (integral) = %f.\n", sum_sim);
    printf("Sum of sins average time on CPU: %f ms\n", elapsed_avg);
}
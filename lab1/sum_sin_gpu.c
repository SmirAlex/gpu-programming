#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FLP_TYPE double

#define N 10000000
const int NUM_ITERATIONS = 5;

 double get_time_ms(struct timespec start, struct timespec end) {
	return (end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec)) * 1000;
 }

FLP_TYPE compute_sum_sin_gpu() {
    FLP_TYPE* sin_arr = (FLP_TYPE*) malloc(sizeof(FLP_TYPE) * N);
    FLP_TYPE step = 2 * M_PI / (FLP_TYPE) (N - 1);
    FLP_TYPE sum_sin = 0.0;

    #pragma acc data create(sin_arr[0:N]) 
    {

    #pragma acc kernels

        for (int i = 0; i < N; i++) {
            sin_arr[i] = sin(i * step);
        }    

    #pragma acc kernels loop reduction(+:sum_sin)
        
        for (int i = 0; i < N; i++) {
            sum_sin += sin_arr[i];
        }
    }
    
    free(sin_arr);
    return sum_sin;
}

int main() {
    struct timespec start, end;
    double total_elapsed = 0;
    FLP_TYPE sum_sim = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        sum_sim = compute_sum_sin_gpu();
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        total_elapsed += get_time_ms(start, end);
    }
    double elapsed_avg = total_elapsed / NUM_ITERATIONS;
    printf("Sum of sins (integral) = %f.\n", sum_sim);
    printf("Sum of sins average time on GPU: %f ms\n", elapsed_avg);
    return 0;
}

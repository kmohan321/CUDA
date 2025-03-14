#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define n 24

int main()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int j;
    float *x, *y;
    x = (float *)malloc(sizeof(float) * n);
    y = (float *)malloc(sizeof(float) * n);
    for (j = 0; j < n; ++j)
    {
        x[j] = (float)1;
        y[j] = (float)1;
    }

    printf("\nx:\n");
    for (j = 0; j < n; ++j)
    {
        printf("%f ", x[j]);
    }

    printf("\ny:\n");
    for (j = 0; j < n; ++j)
    {
        printf("%f ", y[j]);
    }

    float *d_x, *d_y;
    cudaStat = cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaStat = cudaMalloc((void **)&d_y, n * sizeof(float));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
    stat = cublasSetVector(n, sizeof(float), y, 1, d_y, 1);
    float a = 3.0;

    float result;
    stat = cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
    printf("\ndot product x . y : \n ");
    printf (" %7.0f \n " , result );


    cudaFree(d_y);
    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    free(y);
    return 0;
}



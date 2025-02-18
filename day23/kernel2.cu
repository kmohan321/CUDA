#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define n 10

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
        x[j] = (float)j;
        y[j] = (float)j + 1;
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

    stat = cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);
    stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);

    printf("\nNew y:\n");
    for (j = 0; j < n; ++j)
    {
        printf("%f ", y[j]);
    }
    cudaFree(d_y);
    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    free(y);
    return 0;
}
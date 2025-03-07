#include<stdio.h>
#include<cuda.h>

#define BM 32
#define BK 32
#define BN 32
#define TM 4
#define TN 4

__global__ void matmulKernel(float *A, float *B, float *C, int M, int N, int K, int strideA, int strideB) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;
  int innerRowA = threadRow * TM;
  int innerColA = threadCol;
  int innerRowB = threadRow;
  int innerColB = threadCol * TN;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
      for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
          As[(innerRowA + loadOffset) * BK + innerColA] =
              A[(innerRowA + loadOffset) * K + innerColA];
      }
      for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
          Bs[(innerRowB + loadOffset) * BN + innerColB] =
              B[(innerRowB + loadOffset) * N + innerColB];
      }
      __syncthreads();

      A += BK;
      B += BK * N;

      for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
          for (int i = 0; i < TM; ++i) {
              regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
          }
          for (int i = 0; i < TN; ++i) {
              regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
          }
          for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
              for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                  threadResults[resIdxM * TN + resIdxN] +=
                      regM[resIdxM] * regN[resIdxN];
              }
          }
      }
      __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          C[(blockIdx.y * BM + threadRow * TM + resIdxM) * N +
            blockIdx.x * BN + threadCol * TN + resIdxN] =
              threadResults[resIdxM * TN + resIdxN];
      }
  }
}
#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>
#include <time.h>

#define DataType double
#define TPB 32
#define EPS 0.00001

__global__ void matMul(DataType* A, DataType* B, DataType* C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        DataType sum = 0.0;
        for (int k = 0; k < numAColumns; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

DataType randData(DataType min, DataType max) {
  DataType range = (max - min);
  DataType div = RAND_MAX / range;
  return min + (rand() / div);
}

int main(int argc, char** argv) {

    if (argc != 3) {
      printf("expected three arguments, found %d\n", argc);
      exit(-1);
    }

    DataType* hostInputA;
    DataType* hostInputB;
    DataType* hostOutput;
    DataType* resultRef;
    DataType* deviceInputA;
    DataType* deviceInputB;
    DataType* deviceOutput;

    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B

    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = numAColumns;
    numBColumns = numARows;

    printf("%d,", numARows * numAColumns);

    //@@ Insert code below to allocate Host memory for input and output
    hostInputA = (DataType*)malloc(sizeof(DataType) * numARows * numAColumns);
    hostInputB = (DataType*)malloc(sizeof(DataType) * numBRows * numBColumns);
    hostOutput = (DataType*)malloc(sizeof(DataType) * numARows * numBColumns);

    //@@ Insert code below to initialize hostInputA and hostInputB to random numbers, and create reference result in CPU
    srand(time(0));
    resultRef = (DataType*)malloc(sizeof(DataType) * numARows * numBColumns);
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            hostOutput[i * numBColumns + j] = 0;
            for (int k = 0; k < numAColumns; k++) {
                hostInputA[i * numAColumns + k] = randData(0, 100);
                hostInputB[k * numBColumns + j] = randData(0, 100);
                resultRef[i * numBColumns + j] += hostInputA[i * numAColumns + k] * hostInputB[k * numBColumns + j];
            }
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInputA, sizeof(DataType) * numARows * numAColumns);
    cudaMalloc(&deviceInputB, sizeof(DataType) * numBRows * numBColumns);
    cudaMalloc(&deviceOutput, sizeof(DataType) * numARows * numBColumns);

    //@@ Insert code to below to Copy memory to the GPU here
    clock_t start = clock();
    cudaMemcpy(deviceInputA, hostInputA, sizeof(DataType) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInputB, hostInputB, sizeof(DataType) * numBRows * numBColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, sizeof(DataType) * numARows * numBColumns, cudaMemcpyHostToDevice);
    printf("%ld,", clock() - start);

    //@@ Initialize the 2D grid and block dimensions here
    dim3 threadsPerBlock(TPB, TPB);
    dim3 blocksPerGrid((numBColumns + TPB - 1) / TPB, (numARows + TPB - 1) / TPB);

    //@@ Launch the GPU Kernel here
    start = clock();
    matMul<<<blocksPerGrid, threadsPerBlock>>>(deviceInputA, deviceInputB, deviceOutput, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    printf("%ld,", clock() - start);

    //@@ Copy the GPU memory back to the CPU here
    start = clock();
    cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * numARows * numBColumns, cudaMemcpyDeviceToHost);
    printf("%ld,\n", clock() - start);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < numARows * numBColumns; i++) {
      if (abs(resultRef[i] - hostOutput[i]) > EPS) {
        printf("Comparison failed at index %d. Expected %f but got %f.\n", i, resultRef[i], hostOutput[i]);
        exit(-1);
      }
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInputA);
    cudaFree(deviceInputB);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInputA);
    free(hostInputB);
    free(hostOutput);
    free(resultRef);

    return 0;
}


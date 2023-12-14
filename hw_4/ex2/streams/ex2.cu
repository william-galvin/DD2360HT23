/*
Usage: 
$ make
$ ./ex1 <size>
*/


#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>
#include <time.h> 

#define DataType float
#define TPB 32
#define EPS 0.00001
// #define S_SEG (2 << 16)
#define N_STREAMS 4

__global__ void vecAdd(DataType* in1, DataType* in2, DataType* out, int len) {
  uint index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= len) {
    return;
  }
  out[index] = in1[index] + in2[index];
}

DataType randData(DataType min, DataType max) {
  DataType range = (max - min); 
  DataType div = RAND_MAX / range;
  return min + (rand() / div);
}

int main(int argc, char** argv) {

  if (argc != 3) {
    printf("expected two arguments, found %d\n", argc);
    exit(-1);
  }
  
  int inputLength;
  DataType* hostInput1;
  DataType* hostInput2;
  DataType* hostOutput;
  DataType* resultRef;
  DataType* deviceInput1;
  DataType* deviceInput2;
  DataType* deviceOutput;

  inputLength = atoi(argv[1]);
  int S_SEG = atoi(argv[2]);
    
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(sizeof(DataType) * inputLength);
  hostInput2 = (DataType*) malloc(sizeof(DataType) * inputLength);
  hostOutput = (DataType*) malloc(sizeof(DataType) * inputLength);
  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(0));
  resultRef =  (DataType*) malloc(sizeof(DataType) * inputLength);
  for (int i = 0; i < inputLength; i++) {
    hostOutput[i] = 0;
    hostInput1[i] = randData(0, 100);
    hostInput2[i] = randData(0, 100);
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

  clock_t start = clock();

  cudaStream_t streams[N_STREAMS];
  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  for (int i = 0; i < ceil(((double)inputLength) / S_SEG); i++) {
    int offset = i * S_SEG;
    int stream_idx = i % N_STREAMS;
    int bytes = (offset + S_SEG < inputLength ? S_SEG : inputLength - offset) * sizeof(DataType);

    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], bytes, cudaMemcpyHostToDevice, streams[stream_idx]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], bytes, cudaMemcpyHostToDevice, streams[stream_idx]);
    cudaMemcpyAsync(&deviceOutput[offset], &hostOutput[offset], bytes, cudaMemcpyHostToDevice, streams[stream_idx]);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < ceil(((double)inputLength) / S_SEG); i++) {
    int offset = i * S_SEG;
    int stream_idx = i % N_STREAMS;

    dim3 threadsPerBlock(TPB);
    dim3 blocksPerGrid((S_SEG + TPB - 1) / TPB);
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[stream_idx]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], inputLength - offset);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
  }

  for (int i = 0; i < ceil(((double)inputLength) / S_SEG); i++) {
    int offset = i * S_SEG;
    int stream_idx = i % N_STREAMS;
    int bytes = (offset + S_SEG < inputLength ? S_SEG : inputLength - offset) * sizeof(DataType);

    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], bytes, cudaMemcpyDeviceToHost, streams[stream_idx]);
  }

  fprintf(stderr, "total time: ");
  fprintf(stdout, "%ld\n", clock() - start);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++) {
    if (abs(resultRef[i] - hostOutput[i]) > EPS) {
      printf("Comparison failed at index %d. Expected %f but got %f.\n", i, resultRef[i], hostOutput[i]);
      exit(-1);
    }
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  
  return 0;
}

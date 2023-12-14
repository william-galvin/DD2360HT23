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

  if (argc != 2) {
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

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);


  //@@ Initialize the 1D grid and block dimensions here
  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGrid((inputLength + TPB - 1) / TPB);


  //@@ Launch the GPU Kernel here
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);

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

  
  return 0;
}

/* 
  Histogram implementation using streams
  - overlap computation and data transfer

  Usage: 
   $ make
   $ ./ex1 <n>
   $ compute-sanitizer ./ex1 <n>
   $ /usr/local/cuda-11.1/bin/nv-nsight-cu-cli ./ex1 <n>
*/

#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <cstdlib>

#define NUM_BINS 4096
#define TPB 32
#define N_STREAMS 8

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins,
                                 unsigned int offset) {

  uint index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index + offset >= num_elements) {
    return;
  }
  atomicAdd(&bins[input[index]], 1);
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins, unsigned int offset) {
  uint index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index + offset >= num_bins || bins[index] <= 127) {
    return;
  }
  bins[index] = 127;
}


uint randInt() {
  uint div = RAND_MAX / NUM_BINS;
  return (rand() / div);
}


int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("expected two arguments, found %d\n", argc);
    exit(-1);
  }
  inputLength = atoi(argv[1]);

  fprintf(stderr, "The input length is %d\n", inputLength);

  // @@ Insert code below to allocate Host memory for input and output
  hostInput = (uint*) calloc(inputLength, sizeof(uint));
  hostBins = (uint*) calloc(NUM_BINS, sizeof(uint));
  resultRef = (uint*) calloc(NUM_BINS, sizeof(uint));


  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = randInt();
  }

  //@@ Insert code below to create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    resultRef[hostInput[i]]++;
    resultRef[hostInput[i]] = min(resultRef[hostInput[i]], 127);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(uint));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(uint));



  // initialize streams
  const int inputStreamSize = inputLength / N_STREAMS;
  const int inputStreamBytes = inputStreamSize * sizeof(uint);

  const int binsStreamSize = NUM_BINS / N_STREAMS;
  const int binsStreamBytes = binsStreamSize * sizeof(uint);

  cudaStream_t stream[N_STREAMS];
  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamCreate(&stream[i]);
  }


  //@@ Insert code to Copy memory to the GPU here
  //@@ Insert code to initialize GPU results
  clock_t start = clock();
  
  for (int i = 0; i < N_STREAMS; i++) {
    int inputOffset = i * inputStreamSize;
    int binsOffset = i * binsStreamSize;

    cudaMemcpyAsync(
      &deviceInput[inputOffset], // dst
      &hostInput[inputOffset],   // src
      inputStreamBytes,          // count
      cudaMemcpyHostToDevice,    // kind
      stream[i]                  // stream
    );

    cudaMemcpyAsync(
      &deviceBins[binsOffset],   // dst
      &hostBins[binsOffset],     // src
      binsStreamBytes,           // count
      cudaMemcpyHostToDevice,    // kind
      stream[i]                  // stream
    );
  }
  cudaDeviceSynchronize();
  printf("Copy host => device: %ld\n", clock() - start);


  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGridHistogram((inputStreamSize + TPB - 1) / TPB);
  dim3 blocksPerGridConvert((binsStreamSize + TPB - 1) / TPB);

  //@@ Launch the GPU Kernel here
  start = clock();

  for (int i = 0; i < N_STREAMS; i++) {
    int inputOffset = i * inputStreamSize;
    int binsOffset = i * binsStreamSize;

    histogram_kernel<<<
      blocksPerGridHistogram, 
      threadsPerBlock, 
      0, 
      stream[i]
    >>>(&deviceInput[inputOffset], deviceBins, inputLength, NUM_BINS, inputOffset);

    convert_kernel<<<
      blocksPerGridConvert, 
      threadsPerBlock,
      0,
      stream[i]
    >>>(&deviceBins[binsOffset], NUM_BINS, binsOffset);
  }
  cudaDeviceSynchronize();

  printf("both kernels: %ld\n", clock() - start);


  //@@ Copy the GPU memory back to the CPU here
  start = clock();

  for (int i = 0; i < N_STREAMS; i++) {
    int inputOffset = i * inputStreamSize;
    int binsOffset = i * binsStreamSize;

    cudaMemcpyAsync(
      &hostInput[inputOffset], // dst
      &deviceInput[inputOffset],   // src
      inputStreamBytes,          // count
      cudaMemcpyDeviceToHost,    // kind
      stream[i]                  // stream
    );

    cudaMemcpyAsync(
      &hostBins[binsOffset],   // dst
      &deviceBins[binsOffset],     // src
      binsStreamBytes,           // count
      cudaMemcpyDeviceToHost,    // kind
      stream[i]                  // stream
    );
  }
  cudaDeviceSynchronize();

  printf("Copy device => host: %ld\n", clock() - start);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      printf("Error at index %d: expected %d but got %d.\n", i, resultRef[i], hostInput[i]);
      return -1;
    }
  }
  fprintf(stderr, "Output is as expected.\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);


  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  // Clean up streams
  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamDestroy(stream[i]);
  }

  return 0;
}
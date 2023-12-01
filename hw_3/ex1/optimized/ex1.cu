/* 
  Packing histogram implementation
  - input and bins become the same array

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

// CUDA doesn't natively support atomicAdd for uint_16
// need to do some address black magic
// https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712
__device__ short atomicAddShort(uint16_t* address, short val) {
  unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
  unsigned int long_old = atomicAdd(base_address, long_val);
  if((size_t)address & 2) {
    return (short)(long_old >> 16);
  } else {
    unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
    if (overflow)
      atomicSub(base_address, overflow);
      return (short)(long_old & 0xffff);
    }
}

__global__ void histogram_kernel(uint16_t *input, uint16_t *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  uint index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  atomicAddShort(&bins[input[index]], 1);

  if (bins[input[index]] > 127) {
    bins[input[index]] = 127;
  }
}

uint randInt() {
  uint div = RAND_MAX / (NUM_BINS - 1);
  return (rand() / div);
}


int main(int argc, char **argv) {

  int inputLength;
  uint16_t *hostPack;
  uint16_t *resultRef;
  uint16_t *devicePack;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("expected two arguments, found %d\n", argc);
    exit(-1);
  }
  inputLength = atoi(argv[1]);

  fprintf(stderr, "The input length is %d\n", inputLength);

  // @@ Insert code below to allocate Host memory for input and output
  hostPack = (uint16_t*) calloc(inputLength + NUM_BINS, sizeof(uint16_t));
  resultRef = (uint16_t*) calloc(NUM_BINS, sizeof(uint16_t));


  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; i++) {
    hostPack[i] = randInt();
  }

  //@@ Insert code below to create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    resultRef[hostPack[i]]++;
    resultRef[hostPack[i]] = min(resultRef[hostPack[i]], 127);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&devicePack, (inputLength + NUM_BINS) * sizeof(uint16_t));

  clock_t start = clock();
  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(devicePack, hostPack, (inputLength + NUM_BINS) * sizeof(uint16_t), cudaMemcpyHostToDevice);
  printf("Copy host => device: %ld\n", clock() - start);

  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGrid((inputLength + TPB - 1) / TPB);

  //@@ Launch the GPU Kernel here
  start = clock();
  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(devicePack, &devicePack[inputLength], inputLength, NUM_BINS);
  printf("kernel 1: %ld\n", clock() - start);


  //@@ Copy the GPU memory back to the CPU here
  start = clock();
  cudaMemcpy(&hostPack[inputLength], &devicePack[inputLength], NUM_BINS * sizeof(uint16_t), cudaMemcpyDeviceToHost);
  printf("Copy device => host: %ld\n", clock() - start);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostPack[inputLength + i] != resultRef[i]) {
      printf("Error at index %d: expected %d but got %d.\n", i, resultRef[i], hostPack[inputLength + i]);
      return -1;
    }
  }
  fprintf(stderr, "Output is as expected.\n");

  //@@ Free the GPU memory here
  cudaFree(devicePack);

  //@@ Free the CPU memory here
  free(hostPack);
  free(resultRef);

  return 0;
}
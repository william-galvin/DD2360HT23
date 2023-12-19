#include <iostream>

__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) {
    // Calculate the row # of the d_Pin and d_Pout element to process
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column # of the d_Pin and d_Pout element to process
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of d_Pout if in range
    if ((Row < m) && (Col < n)) {
        d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
    }
}

int main() {
    // Specify the size of the picture
    int width = 1024;  // X pixels
    int height = 768;  // Y pixels
    int numElements = width * height;

    // Size in bytes of the picture data
    size_t size = numElements * sizeof(float);

    // Allocating host memory
    float* h_Pin = new float[numElements];
    float* h_Pout = new float[numElements];

    // Initialize input data
    for (int i = 0; i < numElements; ++i) {
        h_Pin[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_Pin;
    float* d_Pout;

    cudaMalloc((void**)&d_Pin, size);
    cudaMalloc((void**)&d_Pout, size);

    // Copy host input data to device
    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);  // 32x32 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    PictureKernel<<<gridSize, blockSize>>>(d_Pin, d_Pout, width, height);
    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    // Check the results
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        // Check if each element in h_Pout is equal to 2 times the corresponding element in h_Pin
        if (std::fabs(h_Pout[i] - 2 * h_Pin[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Passed" << std::endl;
    } else {
        std::cout << "Failed" << std::endl;
    }

    cudaFree(d_Pin);
    cudaFree(d_Pout);
    delete[] h_Pin;
    delete[] h_Pout;

    return 0;
}

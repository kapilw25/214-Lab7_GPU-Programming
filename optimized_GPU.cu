#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

const int N = 131072;

__global__ void optimizedReduction(int* input, int* output, int size) {
    extern __shared__ int sharedData[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        sharedData[threadIdx.x] = input[tid];
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sharedData[threadIdx.x] += sharedData[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[blockIdx.x] = sharedData[0];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <grid_size> <block_size>" << std::endl;
        return 1;
    }

    int gridSize = std::atoi(argv[1]);
    int blockSize = std::atoi(argv[2]);

    int* data;
    int* results;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&results, gridSize * sizeof(int));

    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    optimizedReduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(data, results, N);
    cudaDeviceSynchronize();
    int sum = 0;
    for (int i = 0; i < gridSize; i++) {
        sum += results[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Optimized GPU Sum: " << sum << " Time: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(data);
    cudaFree(results);

    return 0;
}

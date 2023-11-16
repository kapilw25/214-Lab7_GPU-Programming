#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib> // for std::atoi

const int N = 131072;  // Define N here

__global__ void optimizedReductionWarpShuffle(int* input, int* output, int size) {
    extern __shared__ int sharedData[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        sharedData[threadIdx.x] = input[tid];
        __syncthreads();

        // Loop unrolling for the last warp
        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (threadIdx.x < s) {
                sharedData[threadIdx.x] += sharedData[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            // Warp shuffle for final warp reduction
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sharedData[threadIdx.x] += __shfl_down_sync(0xffffffff, sharedData[threadIdx.x], offset);
            }
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
    optimizedReductionWarpShuffle<<<gridSize, blockSize, blockSize * sizeof(int)>>>(data, results, N);
    cudaDeviceSynchronize();
    int sum = 0;
    for (int i = 0; i < gridSize; i++) {
        sum += results[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "optimized_GPU_Reduction_Warp_Shuffle Sum: " << sum << " Time: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(data);
    cudaFree(results);

    return 0;
}

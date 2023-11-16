#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib> // for std::atoi

const int N = 131072;

__global__ void naiveReduction(int* input, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < size; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            input[tid] += input[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = input[0];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./naive <grid_size> <block_size>" << std::endl;
        return 1;
    }

    int gridSize = std::atoi(argv[1]);
    int blockSize = std::atoi(argv[2]);

    int* data;
    int* result;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&result, sizeof(int));

    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    naiveReduction<<<gridSize, blockSize>>>(data, result, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Naive GPU Sum: " << *result << " Time: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(data);
    cudaFree(result);

    return 0;
}

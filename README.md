# GPU Reduction Algorithm Performance Analysis

This repository contains a set of CUDA programs that implement different optimization strategies for the Reduction algorithm on GPUs. The optimizations include naive execution, optimized thread organization, loop unrolling, and warp shuffle instructions.

## Getting Started

These instructions will guide you through the compilation and execution process to generate execution time results for the different GPU conditions.

### Prerequisites
Before you begin, make sure you have the following installed:
- NVIDIA CUDA Toolkit: You can download it from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads) and follow the installation guide for your operating system.
- A CUDA-capable GPU

### Compiling the Programs

Run the following command to compile all the `.cu` files into their respective executables:

```bash
make
```

This will create four executables:

```bash
naive_GPU
optimized_GPU
optimized_GPU_Reduction_Loop_Unroll
optimized_GPU_Reduction_Warp_Shuffle
```

# Generating Execution Times
To run the executables and generate the execution times for specific grid and block sizes, execute the following command:

```bash
make run
```
This will execute each program with grid sizes of 32, 64, and 128, and a block size of 1024, and print the sum and execution times to the console.

# Results
The execution times will be displayed in the terminal window. Here is what the output looks like:


```bash
Running naive_GPU with grid size 32 and block size 1024
Naive GPU Sum: 86167560 Time: 0.00155114 seconds
...
Running optimized_GPU_Reduction_Warp_Shuffle with grid size 128 and block size 1024
optimized_GPU_Reduction_Warp_Shuffle Sum: -1081344 Time: 0.000481708 seconds
```


The Time: field indicates the execution time for the particular optimization strategy and configuration.

# Cleaning Up
To clean up the compiled executables, you can run:


```bash
make clean
```
This will remove all the executables from your directory.

# Authors
Indranil Dutta,
Kapil Wanaskar, 
Prof. Haonan Wang

# Acknowledgments
- San José State University's Computer Engineering Department for providing the necessary resources.
- NVIDIA Corporation for their comprehensive documentation on CUDA development.

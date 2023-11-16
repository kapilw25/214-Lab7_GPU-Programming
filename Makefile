# Define the C++ compiler to use
NVCC = nvcc

# Define any compile-time flags
CFLAGS = -O3

# Define the executable file names
EXECUTABLES = naive_GPU optimized_GPU optimized_GPU_Reduction_Loop_Unroll optimized_GPU_Reduction_Warp_Shuffle

# Define the grid and block sizes to use when running the executables
GRID_SIZES = 32 64 128
BLOCK_SIZE = 1024

.PHONY: all clean run

all: $(EXECUTABLES)

$(EXECUTABLES):
	$(NVCC) $(CFLAGS) $@.cu -o $@

run:
	@for exec in $(EXECUTABLES); do \
		for grid in $(GRID_SIZES); do \
			echo Running $$exec with grid size $$grid and block size $(BLOCK_SIZE); \
			./$$exec $$grid $(BLOCK_SIZE); \
		done; \
	done

clean:
	rm -f $(EXECUTABLES)

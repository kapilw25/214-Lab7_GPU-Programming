import subprocess

# Grid and block sizes
grid_sizes = [2**i for i in range(8)]  # 1, 2, 4, ..., 128
block_sizes = [32, 64, 128, 256, 512, 1024]

# Functions to call and execute the C++/CUDA executables
def run_executable(exec_name, grid_size=1, block_size=32):
    cmd = [exec_name, str(grid_size), str(block_size)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    for line in lines:
        if "Time:" in line:
            return float(line.split()[-2])  # Extract time value

# Serial execution
# serial_time = run_executable("./serial")

# GPU executions
times_naive = []
times_optimized = []
times_loop_unroll = []
times_warp_shuffle = []

for grid in grid_sizes:
    t_n = []
    t_o = []
    t_lu = []  # Initialize list for loop unroll times
    t_ws = []  # Initialize list for warp shuffle times
    for block in block_sizes:
        if grid * block > 131072:  # Adjust if N changes
            continue
        t_n.append(run_executable("./naive", grid, block))
        t_o.append(run_executable("./optimized", grid, block))
        t_lu.append(run_executable("./optimized_loop_unroll", grid, block))  # Assuming executable name
        t_ws.append(run_executable("./optimized_warp_shuffle", grid, block))  # Assuming executable name
    times_naive.append(t_n)
    times_optimized.append(t_o)
    times_loop_unroll.append(t_lu)
    times_warp_shuffle.append(t_ws)

# Save the data into a CSV file
with open('performance_data.csv', 'w') as file:
    file.write('GridSize,BlockSize,Naive_GPU,Optimized_GPU,\
    optimized_GPU_Reduction_Loop_Unroll,\
    optimized_GPU_Reduction_Warp_Shuffle\n')
    for i, grid_size in enumerate(grid_sizes):
        for j, block_size in enumerate(block_sizes):
            if j < len(times_naive[i]):
                file.write(f"{grid_size},{block_size},{times_naive[i][j]},\
                {times_optimized[i][j]},{times_loop_unroll[i][j]},\
                {times_warp_shuffle[i][j]}\n")

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('performance_data.csv')

# Calculate the composite size as a string of 'GridSize*BlockSize'
df['CompositeSize'] = (df['GridSize'].astype(str) + "*" + df['BlockSize'].astype(str))

# Plot the data
plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

# Plot each metric
plt.plot(df['CompositeSize'], df['Naive_GPU'], label='Naive_GPU', marker='o')
plt.plot(df['CompositeSize'], df['Optimized_GPU'], label='Optimized_GPU', marker='o')

# Adjusted to match the column names exactly (including spaces)
plt.plot(df['CompositeSize'], df['    optimized_GPU_Reduction_Loop_Unroll'], label='Optimized_GPU_Loop_Unroll', marker='x')
plt.plot(df['CompositeSize'], df['    optimized_GPU_Reduction_Warp_Shuffle'], label='Optimized_GPU_Warp_Shuffle', marker='^')

# Uncomment and adjust if Serial_CPU data is present
# plt.plot(df['CompositeSize'], df['Serial_CPU'], label='Serial_CPU', marker='o', linestyle='--')

# Set the x-axis labels with a rotation for better readability
plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Composite Size (GridSize*BlockSize)')
plt.ylabel('Execution Time (s)')
plt.title('Performance Comparison Across Different GPU Optimizations')

# Adding a legend to the plot
plt.legend()

# Ensure layout fits the labels
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('performance_plot.png', dpi=300)

# Close the plot after saving to avoid displaying it in an interactive window
plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (update the filename)
df = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/speedy_v100_b4_50epochs_offandon.csv")  # Replace with actual file



# Load your data

# Ensure all columns are numeric (if necessary)
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Group by epoch and iteration
df_avg = df.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time':'min',
}).reset_index()





# Read the CSV file, skipping the first row which contains repeated column headers
df2 = pd.read_csv("/dl-bench/rnouaj/mlcomns_imseg/pytorch_v100_b4_4GPUs_pytorch_50ep.csv")


# Ensure all columns are numeric (if necessary)
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

# Group by epoch and iteration
df_avg2 = df2.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time':'min',
}).reset_index()



df1 = pd.read_csv("/home/2023/rnouaj/Baselines_Speedy/mlcomns_imseg_with_dali/pytorch_v100_b4_4GPUs_dali_50epochs.csv")  # Replace with actual file

# Compute the average for each iteration
# Ensure all columns are numeric (if necessary)
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

# Group by epoch and iteration
df_avg1 = df1.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time':'min',
}).reset_index()

#add 100 to iteration time of df_avg1 after epoch 2
for index, row in df_avg1.iterrows():
    if row['epoch'] > 2:
        df_avg1.at[index, 'iteration_time'] += 50


# Plot the results
plt.figure(figsize=(15, 8))
plt.plot(df_avg['iteration_time'], df_avg['throughput(MBs)'], marker='o', linestyle='-', label='speedy')
plt.plot(df_avg2['iteration_time'], df_avg2['throughput(MBs)'], marker='o', linestyle='-', label='Pytorch')
plt.plot(df_avg1['iteration_time'], df_avg1['throughput(MBs)'], marker='o', linestyle='-', label='Dali')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Throughput (MB/s)")
plt.title("Throughput MB/s 3d-UNet 50 epochs V100")
plt.savefig("throughput_all_except_pecan_3dunet.png")  # Save the plot to a file
plt.show()

import csv
import numpy as np
import ast  # Safely parse the batch_composition string

# Lists to hold counts per batch
s_counts = []
l_counts = []

# Read the CSV
with open('batch_composition.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        composition = ast.literal_eval(row['batch_composition'])  # e.g., ('S', 'S', 'S', 'S')
        s_count = composition.count('S')
        l_count = composition.count('L')
        s_counts.append(s_count)
        l_counts.append(l_count)

# Calculate average and std
s_avg = np.mean(s_counts)
s_std = np.std(s_counts)
l_avg = np.mean(l_counts)
l_std = np.std(l_counts)

print("Average 'S' per batch: {:.2f}, Std Dev: {:.2f}".format(s_avg, s_std))
print("Average 'L' per batch: {:.2f}, Std Dev: {:.2f}".format(l_avg, l_std))

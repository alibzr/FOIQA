import os
import numpy as np
import pandas as pd
from skimage.io import imread
from joblib import Parallel, delayed
from tqdm import tqdm
import natsort

# Define the directory containing the files
directory = "masked_fixmaps"

# Function to process each file and count non-zero pixels
def process_file(file):
    # Read the image
    image = imread(os.path.join(directory, file))
    # Count non-zero pixels
    non_zero_count = np.count_nonzero(image)
    return non_zero_count

# Get a list of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.png')]
files = natsort.natsorted(files)

# Extract unique combinations of session, scene, and distortion
unique_combinations = list(set(['-'.join(f.split('-')[1:4]) for f in files]))

# Initialize a dictionary to store non-zero counts for each combination
results = {comb: [] for comb in unique_combinations}

# Process files in parallel using Joblib
non_zero_counts = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in tqdm(files))

# Group non-zero counts by unique combinations
for file, count in zip(files, non_zero_counts):
    combination = '-'.join(file.split('-')[1:4])
    results[combination].append(count)

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame.from_dict(results, orient='index').transpose()
df.to_csv('sorted_non_zero_counts.csv', index=False)

print("Processing complete. Results saved to non_zero_counts.csv.")

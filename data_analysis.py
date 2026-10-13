import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

filter = ["Protocol Type", "Rate", "ack_count", "syn_count", "rst_count",
    "Tot sum", "Min", "Max", "Std", "AVG", "IAT", "Label"]

csv_files = glob.glob('CIC_IOT_Dataset2023/*.csv')
file_chunks = []
for file in csv_files:
    with pd.read_csv(file, chunksize = 100000) as reader:
        for chunk in reader:
            chunk = chunk[filter]
            file_chunks.append(chunk)
df = pd.concat(file_chunks, ignore_index=True)

#check for bad values
print("col: inf | nan")
for col in df.select_dtypes(include=np.number).columns:
    print(f"{col}: {np.isinf(df[col]).sum()} | {np.isnan(df[col]).sum()}")

#get distribution of labels and plot
print(df["Label"].value_counts())

def plot_distribution(value_counts):
    plt.figure(figsize=(10, 5))
    value_counts.plot(kind="bar", color="skyblue")
    plt.title("Distribution of Values")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()

plot_distribution(df["Label"].value_counts(ascending = True))
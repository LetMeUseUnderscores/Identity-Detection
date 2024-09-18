import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Data\\FairFace\\train_labels.csv', header=None)

# Find the number of unique values in column 4 (Python uses 0-based indexing)
num_unique_values = df[3].nunique()

print(f'Number of unique values in column 4: {num_unique_values}')
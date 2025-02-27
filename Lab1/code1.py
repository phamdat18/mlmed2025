import pandas as pd

# Load the first CSV file
df1 = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\mitbih_test.csv')

# Load the second CSV file
df2 = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\mitbih_train.csv')

# Concatenate the two DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv('concatenated_file1.csv', index=False)

# Check for missing values in the dataset
missing_data = concatenated_df.isnull().sum()

# Plot histograms for a few columns to examine the distribution of values
import matplotlib.pyplot as plt

# Plot histogram for a few selected columns
columns_to_plot = concatenated_df.columns[:5]  # Just taking the first 5 columns for illustration
plt.figure(figsize=(10, 6))
for i, col in enumerate(columns_to_plot):
    plt.subplot(2, 3, i+1)
    concatenated_df[col].hist(bins=50, ax=plt.gca())
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Check the percentage of missing values for each column
missing_percentage = (missing_data / len(concatenated_df)) * 100

missing_data.head(), missing_percentage.head()

import pandas as pd

# Load the first CSV file
df1 = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\ptbdb_abnormal.csv')

# Load the second CSV file
df2 = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\ptbdb_normal.csv')

# Concatenate the two DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv('concatenated_file2.csv', index=False)

# Check for missing values in the dataset
missing_data = concatenated_df.isnull().sum()

# Load the newly uploaded file for EDA
df_eda = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\concatenated_file2.csv')

# Check the structure of the dataset
data_shape = df_eda.shape
column_names = df_eda.columns

# Get the summary statistics of the dataset
summary_stats = df_eda.describe()

# Check for missing data
missing_data = df_eda.isnull().sum()

# Display the first few rows of the dataset to get a glimpse of the data
df_eda.head(), data_shape, column_names, summary_stats.head(), missing_data.head()
print(data_shape)

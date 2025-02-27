import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\concatenated_file2.csv')

# Remove rows where the target variable 'y' is NaN
df = df.dropna(subset=[df.columns[-1]])

# Remove features with constant values (i.e., all values are the same)
df = df.loc[:, (df.nunique() > 1)]

# Assuming the last column is the label and all other columns are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Impute missing values in the features with the column mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

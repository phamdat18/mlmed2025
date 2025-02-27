import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your training and test datasets
df_train = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\mitbih_train.csv')
df_test = pd.read_csv(r'E:\PhamTienDat_22BI13080\mlmed2025\Lab1\mitbih_test.csv')

# Assuming the last column is the label and all other columns are features
X_train = df_train.iloc[:, :-1]  
y_train = df_train.iloc[:, -1]   

X_test = df_test.iloc[:, :-1]    
y_test = df_test.iloc[:, -1]     

# Align the columns between the training and test sets
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

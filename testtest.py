from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Load your dataset
data_path = './load_management_dataset_india_500.csv'
data_frame = pd.read_csv(data_path)

# Print the column names to verify
print("Column names:", data_frame.columns)

# Check if the expected columns are present
expected_columns = ['ime_of_day', 'day_of_week', 'season', 'temperature', 'humidity', 'historical_load', 'current_load']
missing_columns = [col for col in expected_columns if col not in data_frame.columns]

if missing_columns:
    raise KeyError(f"The following expected columns are missing from the dataset: {missing_columns}")

# Extract the relevant columns
X = data_frame[expected_columns]

# Extract the target variable
y = data_frame['load_label']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop='first' to avoid multicollinearity

# Apply one-hot encoding to categorical features
encoded_features = encoder.fit_transform(X[['ime_of_day', 'day_of_week', 'season']])

# Combine one-hot encoded features with other numerical features
numerical_features = X[['temperature', 'humidity', 'historical_load', 'current_load']].values
encoded_data = np.hstack((encoded_features, numerical_features))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_data, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'{name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, predictions))

# Cross-validation for more robust results
for name, model in models.items():
    cv_scores = cross_val_score(model, encoded_data, y, cv=5)
    print(f'{name} Cross-Validation Accuracy: {cv_scores.mean():.2f}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
data_frame = pd.read_csv('./load_management_dataset_india_500.csv')

# Extract features and target variable
X = data_frame[['time_of_day', 'day_of_week', 'season', 'temperature', 'humidity', 'historical_load', 'current_load']]
y = data_frame['load_label']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop='first' to avoid multicollinearity

# Apply one-hot encoding to categorical features
encoded_ime_of_day = encoder.fit_transform(X[['time_of_day']])
encoded_day_of_week = encoder.fit_transform(X[['day_of_week']])
encoded_season = encoder.fit_transform(X[['season']])

# Combine one-hot encoded features with other numerical features
numerical_features = X[['temperature', 'humidity', 'historical_load', 'current_load']].values
encoded_data = np.hstack((encoded_ime_of_day, encoded_day_of_week, encoded_season, numerical_features))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(encoded_data, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=100),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
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

print(data_frame.head())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv('building_dataset.csv')

# Define the expected columns and rename them
expected_columns = {
    'Date': 'time_of_day',
    'Room_ID': 'room_id',
    'Number_of_AC_Units': 'number_of_ac_units',
    'Number_of_Fans': 'number_of_fans',
    'Number_of_Lights': 'number_of_lights',
    'Number_of_Projectors': 'number_of_projectors',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Electricity_Consumption': 'historical_load',
    'Load_Label': 'load_label'
}

# Check for missing columns
missing_columns = [col for col in expected_columns.keys() if col not in df.columns]
if missing_columns:
    raise KeyError(f"The following expected columns are missing from the dataset: {missing_columns}")

# Rename columns to match expected names
df.rename(columns=expected_columns, inplace=True)

# Convert the 'Date' column to datetime
df['time_of_day'] = pd.to_datetime(df['time_of_day'])

# Feature engineering: Extract additional features from the datetime
df['day_of_week'] = df['time_of_day'].dt.day_name()
df['hour_of_day'] = df['time_of_day'].dt.hour

# Drop the original 'time_of_day' column
df.drop(columns=['time_of_day'], inplace=True)

# Convert categorical columns to numeric (One-Hot Encoding for day_of_week)
df = pd.get_dummies(df, columns=['day_of_week'])

# Define features and target
X = df.drop(columns=['load_label'])
y = df['load_label']

# Separate numeric features for scaling
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X_numeric = X[numeric_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Standardize the numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))


import joblib

# Save the trained model to a file
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)

# Save the scaler as well
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassif

# Load the saved model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example live input data (you can replace this with real data)
live_data = pd.DataFrame({
    'room_id': [1],
    'number_of_ac_units': [2],
    'number_of_fans': [6],
    'number_of_lights': [6],
    'number_of_projectors': [1],
    'temperature': [22.5],
    'humidity': [45],
    'hour_of_day': [14],
    'day_of_week_Monday': [1],
    'day_of_week_Tuesday': [0],
    'day_of_week_Wednesday': [0],
    'day_of_week_Thursday': [0],
    'day_of_week_Friday': [0],
    'day_of_week_Saturday': [0],
    'day_of_week_Sunday': [0]
})

# Ensure columns match the training data
numeric_features = ['number_of_ac_units', 'number_of_fans', 'number_of_lights', 'number_of_projectors',
                    'temperature', 'humidity', 'hour_of_day', 'day_of_week_Monday', 'day_of_week_Tuesday',
                    'day_of_week_Wednesday', 'day_of_week_Thursday', 'day_of_week_Friday',
                    'day_of_week_Saturday', 'day_of_week_Sunday']
live_data = live_data[numeric_features]

# Scale the live input data
live_data_scaled = scaler.transform(live_data)

# Make predictions
predictions = model.predict(live_data_scaled)

# Print the prediction
print("Prediction:", predictions)
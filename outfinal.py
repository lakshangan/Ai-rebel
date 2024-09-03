import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the new dataset
new_df = pd.read_csv('new_building_dataset.csv')

# Define features and target
X_new = new_df.drop(columns=['Load_Label'])
y_new = new_df['Load_Label']

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('random_forest_model.pkl')

# Standardize the new data
X_new_scaled = scaler.transform(X_new)

# Make predictions
y_new_pred = model.predict(X_new_scaled)

# Add predictions to the dataframe
new_df['predicted_load_label'] = y_new_pred

# Print the peak voltage when it recognizes the peak value
peak_voltage = new_df[new_df['predicted_load_label'] == 'Peak']['historical_load'].max()
print(f"Peak voltage used: {peak_voltage} W")

# Print the count of predictions for each category
print(new_df['predicted_load_label'].value_counts())

# Save the predictions to a CSV file
new_df.to_csv('predicted_new_building_dataset.csv', index=False)



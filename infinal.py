import pandas as pd
import numpy as np

# Load the existing dataset
df = pd.read_csv('building_dataset.csv')

# Set random seed for reproducibility
np.random.seed(42)

# Define the size of the new dataset
n_samples = len(df)

# Generate random data for each feature based on the existing dataset
new_data = {
    'Date': pd.to_datetime(df['Date']) + pd.to_timedelta(np.random.randint(1, 365, size=n_samples), unit='D'),
    'Room_ID': np.random.choice(df['Room_ID'].unique(), size=n_samples),
    'Number_of_AC_Units': np.random.choice(df['Number_of_AC_Units'].unique(), size=n_samples),
    'Number_of_Fans': np.random.choice(df['Number_of_Fans'].unique(), size=n_samples),
    'Number_of_Lights': np.random.choice(df['Number_of_Lights'].unique(), size=n_samples),
    'Number_of_Projectors': np.random.choice(df['Number_of_Projectors'].unique(), size=n_samples),
    'Temperature': np.random.uniform(df['Temperature'].min(), df['Temperature'].max(), size=n_samples),
    'Humidity': np.random.uniform(df['Humidity'].min(), df['Humidity'].max(), size=n_samples),
    'Electricity_Consumption': np.random.uniform(df['Electricity_Consumption'].min(), df['Electricity_Consumption'].max(), size=n_samples),
}

# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# Apply similar conditions for labeling 'Load_Label' (assuming this was the original logic)
conditions = [
    (new_df['Electricity_Consumption'] > 4000),
    (new_df['Electricity_Consumption'] > 3000) & (new_df['Electricity_Consumption'] <= 4000),
    (new_df['Electricity_Consumption'] > 1500) & (new_df['Electricity_Consumption'] <= 3000),
    (new_df['Electricity_Consumption'] <= 1500)
]
labels = ['Peak', 'High', 'Normal', 'Low']

new_df['Load_Label'] = np.select(conditions, labels)

# Save the new dataset to a CSV file
new_df.to_csv('new_building_dataset.csv', index=False)

print("New unseen dataset has been generated and saved as 'new_building_dataset.csv'.")
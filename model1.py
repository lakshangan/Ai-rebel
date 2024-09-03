import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/Users/lakshanganesan/Desktop/train/building_dataset.csv')

# Print the column names for debugging purposes
print("Column names:", df.columns)

# Define the expected columns
expected_columns = ['time_of_day', 'day_of_week', 'season', 'temperature', 'humidity', 'historical_load', 'current_load', 'load_label']

# Check for missing columns
missing_columns = [col for col in expected_columns if col not in df.columns]

# If there are missing columns, raise an error
if missing_columns:
    raise KeyError(f"The following expected columns are missing from the dataset: {missing_columns}")

# Proceed with the rest of your script
print("All expected columns are present.")
# Example operation (add your further processing here)
print(df.head())

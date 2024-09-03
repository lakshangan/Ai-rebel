import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters for the dataset
floors = 4
classes_per_floor = 20
days = 365 * 2  # 2 years of data
base_date = datetime(2022, 1, 1)

# Create lists for each parameter
dates = [base_date + timedelta(days=i) for i in range(days)]
room_ids = [f"Room_{floor:02d}{cls:02d}" for floor in range(1, floors+1) for cls in range(1, classes_per_floor+1)]

# Initialize empty DataFrame
data = []

# Generate data
for date in dates:
    for room_id in room_ids:
        number_of_ac_units = np.random.randint(1, 4)  # Simulate variation in AC usage
        number_of_fans = 6
        number_of_lights = 6
        number_of_projectors = 1
        temperature = np.random.uniform(18, 30)  # Simulate temperature variations
        humidity = np.random.uniform(30, 70)  # Simulate humidity variations
        electricity_consumption = (
            1.5 * number_of_ac_units * temperature +
            0.1 * number_of_fans +
            0.2 * number_of_lights +
            0.3 * number_of_projectors +
            0.01 * humidity +
            np.random.normal(0, 2)
        )
        load_label = 1 if electricity_consumption > 20 else 0  # Simplified threshold for high/low load

        data.append([date, room_id, number_of_ac_units, number_of_fans, number_of_lights,
                     number_of_projectors, temperature, humidity, electricity_consumption, load_label])

# Create DataFrame
columns = ['Date', 'Room_ID', 'Number_of_AC_Units', 'Number_of_Fans', 'Number_of_Lights',
           'Number_of_Projectors', 'Temperature', 'Humidity', 'Electricity_Consumption', 'Load_Label']
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
dataset_path = '/Users/lakshanganesan/Desktop/train/building_dataset.csv'
df.to_csv(dataset_path, index=False)

dataset_path

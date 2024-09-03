import pandas as pd
import numpy as np
import datetime

# Parameters
n_floors = 4
n_rooms_per_floor = 20
total_rooms = n_floors * n_rooms_per_floor
days = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

# Create empty list to store data
data = []

# Generate data
for date in days:
    for room_id in range(1, total_rooms + 1):
        room_str = f"Room_{room_id:03d}"
        temperature = np.random.uniform(20, 25)
        humidity = np.random.uniform(45, 55)
        electricity_consumption = 50 + (6 * 0.5 + 6 * 0.3 + 1 * 1.0) + np.random.uniform(-10, 10)
        load_label = 'High' if electricity_consumption > 75 else 'Low'
        
        data.append([date, room_str, 1, 6, 6, 1, temperature, humidity, electricity_consumption, load_label])

# Create DataFrame
df = pd.DataFrame(data, columns=['Date', 'Room_ID', 'Number_of_AC_Units', 'Number_of_Fans', 'Number_of_Lights', 'Number_of_Projectors', 'Temperature', 'Humidity', 'Electricity_Consumption', 'Load_Label'])

# Save to CSV
df.to_csv('building_dataset.csv', index=False)

import pandas as pd
import random

# Parameters
num_classrooms = 80

data = {
    'Number_of_Fans': [random.randint(4, 6) for _ in range(num_classrooms)],
    'Number_of_Lights': [random.randint(4, 6) for _ in range(num_classrooms)],
    'Number_of_Projectors': [1 for _ in range(num_classrooms)],
    'Number_of_AC_Units': [random.randint(1, 2) for _ in range(num_classrooms)],
    'Number_of_Computers': [random.randint(0, 5) for _ in range(num_classrooms)],
    'Number_of_Hours_Used_Per_Day': [random.randint(5, 10) for _ in range(num_classrooms)],
    'Room_Size (sq. meters)': [random.randint(30, 60) for _ in range(num_classrooms)],
    'Season': [random.choice(['Summer', 'Winter', 'Spring', 'Autumn']) for _ in range(num_classrooms)],
    'Number_of_Students': [random.randint(20, 50) for _ in range(num_classrooms)],
}

# Calculating Total Power Consumption based on the parameters
def calculate_power(row):
    power_fans = row['Number_of_Fans'] * 60
    power_lights = row['Number_of_Lights'] * 10
    power_projector = row['Number_of_Projectors'] * 300
    power_ac = row['Number_of_AC_Units'] * 1500
    power_computers = row['Number_of_Computers'] * 200

    # Adjusting power consumption based on season
    if row['Season'] == 'Summer':
        power_ac *= 1.2  # Increased AC usage in summer
    elif row['Season'] == 'Winter':
        power_ac *= 0.8  # Reduced AC usage in winter

    total_power = (power_fans + power_lights + power_projector + power_ac + power_computers) * row['Number_of_Hours_Used_Per_Day']
    return total_power

# Create the DataFrame
df = pd.DataFrame(data)
df['Total_Power_Consumption (Watts)'] = df.apply(calculate_power, axis=1)

# Save the dataset to a CSV file
df.to_csv('expanded_electricity_consumption_dataset.csv', index=False)

print("Dataset created with more parameters!")

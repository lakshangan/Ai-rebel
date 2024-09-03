import pandas as pd

# Example input data for prediction
input_data = {
    'Number_of_AC_Units': [2],        # Example: 2 AC units in the room
    'Number_of_Fans': [6],            # Fixed: 6 fans in the room
    'Number_of_Lights': [6],          # Fixed: 6 lights in the room
    'Number_of_Projectors': [1],      # Fixed: 1 projector in the room
    'Temperature': [25.0],            # Example: Current temperature is 25°C
    'Humidity': [50.0],               # Example: Current humidity is 50%
    'day_of_week_Monday': [1],        # Set to 1 if Monday; other days should be 0
    'day_of_week_Tuesday': [0],       # Set to 1 if Tuesday; other days should be 0
    'day_of_week_Wednesday': [0],     # Set to 1 if Wednesday; other days should be 0
    'day_of_week_Thursday': [0],      # Set to 1 if Thursday; other days should be 0
    'day_of_week_Friday': [0],        # Set to 1 if Friday; other days should be 0
    'day_of_week_Saturday': [0],      # Set to 1 if Saturday; other days should be 0
    'day_of_week_Sunday': [0],        # Set to 1 if Sunday; other days should be 0
    'hour_of_day': [14],              # Example: Current hour is 2 PM (14:00)
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame(input_data)

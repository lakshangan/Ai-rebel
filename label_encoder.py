import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample dataset
data = pd.DataFrame({
    'ime_of_day': ['night', 'afternoon', 'evening', 'afternoon', 'morning'],
    'day_of_week': ['weekday', 'weekend', 'weekday', 'weekday', 'weekend'],
    'season': ['summer', 'summer', 'autumn', 'spring', 'autumn'],
    'temperature': [43.052578, 41.733532, 22.625522, 33.624184, 29.733281],
    'humidity': [73.713334, 59.688081, 78.324633, 69.484441, 56.476418],
    'historical_load': [443.827017, 474.362988, 576.072634, 492.352631, 440.432868],
    'current_load': [726.442635, 505.416835, 819.371950, 692.769021, 893.552074],
    'load_label': [1, 0, 2, 0, 2]
})

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop='first' to avoid multicollinearity

# Apply one-hot encoding to 'ime_of_day', 'day_of_week', and 'season'
encoded_ime_of_day = encoder.fit_transform(data[['ime_of_day']])
encoded_day_of_week = encoder.fit_transform(data[['day_of_week']])
encoded_season = encoder.fit_transform(data[['season']])

# Combine one-hot encoded features with other numerical features
numerical_features = data[['temperature', 'humidity', 'historical_load', 'current_load']].values
encoded_data = np.hstack((encoded_ime_of_day, encoded_day_of_week, encoded_season, numerical_features))


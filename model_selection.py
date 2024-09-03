from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# data path 
data_path= './load_management_dataset_india_500.csv'

# Load your dataset
data_frame = pd.read_csv(data_path)

# setting the X and y
X = data_frame[[
    "temperature",
    "humidity",
    "historical_load",
    "current_load"
]]

y = data_frame["load_label"]



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'knn': KNeighborsClassifier(),

}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'{name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, predictions))

# Cross-validation for more robust results
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f'{name} Cross-Validation Accuracy: {cv_scores.mean():.2f}')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# User Params
#max_lat = NA
#min_lat = NA

# Load the survey data into a pandas dataframe
survey_data = pd.read_csv("survey_data.csv")

# Select the input features and target variable
X = survey_data[["LATITUDE", "LONGITUDE"]]
y = survey_data["INCOME"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Generate random spatial points
n_points = 100
min_lat, max_lat = X["LATITUDE"].min(), X["LATITUDE"].max()
min_lon, max_lon = X["LONGITUDE"].min(), X["LONGITUDE"].max()
random_lats = np.random.uniform(min_lat, max_lat, n_points)
random_lons = np.random.uniform(min_lon, max_lon, n_points)
new_points = pd.DataFrame({"LATITUDE": random_lats, "LONGITUDE": random_lons})

# Use the trained model to predict the "INCOME" value for the new points
predictions = rfc.predict(new_points)

# Print the predicted "INCOME" values for the new points
print(predictions)
